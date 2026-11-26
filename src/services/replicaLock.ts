import { AsyncLocalStorage } from 'node:async_hooks';
import type { QueryRunner } from 'typeorm';
import { getAppDataSource } from '../db/connection.js';
import { databaseTransaction } from '../db/transactionContext.js';
import { logger } from '../utils/logger.js';
import { isReplicaMode } from '../utils/replicaMode.js';

/**
 * State of a replica lock, seen by the tasks started under it, which inherit it and may run
 * after its release:
 * - `live`: the lock connection holds the advisory lock;
 * - `lost`: the lock could not be taken again after a commit, or the state of the lock
 *   transaction is unknown after a failed savepoint operation; the action still runs, but
 *   nothing that needs the lock may run any more;
 * - `released`: the action has returned.
 */
interface LockOwner {
  runner: QueryRunner;
  state: 'live' | 'lost' | 'released';
}
const heldLocks = new AsyncLocalStorage<Map<string, LockOwner>>();
const heldOwner = (scope: string): LockOwner | undefined => heldLocks.getStore()?.get(scope);
const lockConnectionKey = (scope: string) => JSON.stringify(['lock-connection', scope]);
const lockLostError = () =>
  new Error('Replica lock lost after a failed acquisition; retry the request');
/**
 * Credential work in progress on a lock connection. Each piece of work runs in its own
 * savepoint and opens a frame: nested work started while the frame is active joins it, one
 * piece after the other (savepoints of one connection must nest), and the outermost work
 * commits for all of them (`root`). Tasks started under a frame inherit it and may run after it
 * ends; they then queue like any other caller.
 */
interface LockWork {
  commit: boolean;
}
interface LockWorkFrame {
  root: LockWork;
  active: boolean;
  queue: Promise<unknown>;
}
const lockWork = new AsyncLocalStorage<Map<string, LockWorkFrame>>();
// Wait budget for a replica lock: the in-process queue and the database lock together.
const LOCK_TIMEOUT_MS = 10_000;
/**
 * Bound of the row-lock waits of credential work on the lock connection. A binding write
 * updates the replica revision row (statement trigger), which a configuration write holds until
 * its transaction ends: longer than the acquisition timeout to let configuration writes finish,
 * but finite, so a write waiting for a transaction that waits for it (a configuration request
 * refreshing a personal grant after a configuration write) fails instead of hanging while that
 * request holds the configuration lock of every replica.
 */
const CREDENTIAL_WORK_LOCK_TIMEOUT = '60s';

/** Roll back after `error`; a rollback failure is logged and never hides `error`. */
const rollbackAfter = async (
  runner: QueryRunner,
  error: unknown,
  onFailure?: () => void,
): Promise<never> => {
  try {
    await runner.rollbackTransaction();
  } catch (rollbackError) {
    onFailure?.();
    logger.warn('Replica lock rollback failed', rollbackError);
  }
  throw error;
};

/**
 * Open the lock transaction and take the transaction-level advisory lock. Transaction-level
 * only, so the lock also works through a transaction pooler (PgBouncer in transaction mode):
 * no state outlives a transaction. `timeoutMs` covers the acquisition only; the credential work
 * of the transaction gets {@link CREDENTIAL_WORK_LOCK_TIMEOUT}.
 */
const acquire = async (
  runner: QueryRunner,
  scope: string,
  timeoutMs = LOCK_TIMEOUT_MS,
): Promise<void> => {
  await runner.startTransaction();
  try {
    // SET takes no parameters; the values are an integer and a constant.
    await runner.query(`SET LOCAL lock_timeout = '${Math.max(1, Math.ceil(timeoutMs))}ms'`);
    await runner.query('SELECT pg_advisory_xact_lock(209, hashtext($1))', [scope]);
    await runner.query(`SET LOCAL lock_timeout = '${CREDENTIAL_WORK_LOCK_TIMEOUT}'`);
  } catch (error) {
    await rollbackAfter(runner, error);
  }
};

/**
 * End the lock transaction and return its connection. A plain ROLLBACK ends the transaction
 * whatever its savepoint depth; if it fails, the connection is discarded, so its transaction
 * and advisory lock end with its session instead of going back to the pool.
 */
const endLockConnection = async (runner: QueryRunner): Promise<void> => {
  try {
    if (runner.isTransactionActive) await runner.query('ROLLBACK');
  } catch (error) {
    logger.warn('Replica lock release failed; discarding its connection', error);
    // TypeORM's PostgresQueryRunner hands the error to pg-pool, which destroys the client.
    const discard = (
      runner as unknown as { releasePostgresConnection?: (error: unknown) => Promise<void> }
    ).releasePostgresConnection;
    if (discard) return discard.call(runner, error);
  }
  await runner.release();
};

/**
 * Credential-scoped, reentrant lock across replicas; it never retries the action after a
 * database failure.
 *
 * The action runs in the caller's context (a request transaction included). Callers of this
 * process queue in memory first, without holding a connection; then the lock is a
 * transaction-level advisory lock held by a transaction on a dedicated connection, which also
 * runs the credential work passed to {@link withLockConnection}. Waiting for both fails after
 * 10 seconds. Nested calls reenter while the lock is live, and fail once it is lost; a task
 * started under the lock that calls it after the release queues like any other caller.
 */
export const withReplicaLock = async <T>(scope: string, action: () => Promise<T>): Promise<T> => {
  if (!isReplicaMode()) return action();
  const held = heldOwner(scope);
  // Reentrant only while the lock is live: a task started under it may resume after the release.
  if (held?.state === 'live') return action();
  // Queuing would wait for the action holding this process's turn, which runs this call.
  if (held?.state === 'lost') throw lockLostError();
  // Same-process holders queue here, before the database lock: withLockConnection releases and
  // takes the database lock again after a write, and a request of this process waiting for the
  // in-process lock while holding the database one would block that new acquisition. Not
  // reentrant: an inherited queue marker would let a late task skip the queue.
  const deadline = Date.now() + LOCK_TIMEOUT_MS;
  return withKeyedLock(
    JSON.stringify(['replica-lock', scope]),
    async () => {
      const runner = getAppDataSource().createQueryRunner();
      let owner: LockOwner | undefined;
      try {
        await runner.connect();
        await acquire(runner, scope, deadline - Date.now());
        owner = { runner, state: 'live' };
        return await heldLocks.run(new Map(heldLocks.getStore()).set(scope, owner), action);
      } finally {
        if (owner) owner.state = 'released';
        // Wait for credential work still running on the connection, such as a task started under
        // the lock: committed writes are durable, the rest is rolled back.
        await withKeyedLock(lockConnectionKey(scope), () => endLockConnection(runner), {
          reentrant: false,
        });
      }
    },
    { timeoutMs: LOCK_TIMEOUT_MS, reentrant: false },
  );
};

/**
 * Run credential work on the connection holding the replica lock of `scope`, outside any
 * request transaction, in a savepoint of the lock transaction (rolled back if `work` fails).
 * With `commit`, the lock transaction is committed after `work` and the lock taken again.
 *
 * Use it for credential reads and writes that follow irreversible upstream side effects: a
 * refresh rotates the refresh token at the authorization server, so the new grant must be
 * durable at once (a rolled-back dashboard request must not restore the revoked token, and a
 * long locked action, such as a tool call, must not keep the replica revision row locked by
 * the binding write), and reads must see the latest committed state (a REPEATABLE READ
 * snapshot must not hide a rotation made by another request). The lock connection is already
 * held, so this needs no extra pooled connection. Without a live lock (outside replica mode,
 * or work that waited past the release of the lock), `work` runs in the caller's context, as
 * credential work did before; the binding compare-and-save still rejects stale writes. The
 * release waits for credential work already running on the lock connection. Row-lock waits of
 * `work` fail after {@link CREDENTIAL_WORK_LOCK_TIMEOUT}.
 *
 * Nested work joins the enclosing one while it runs, one piece after the other, and the commit
 * happens when the outermost work ends.
 *
 * Between that commit and the new acquisition, another replica may take the lock: it then
 * reads the committed grant. If it keeps the lock longer than the timeout, the new acquisition
 * fails: the committed work still returns its result, the lock is lost, and the next credential
 * work or nested {@link withReplicaLock} call of this lock fails with "Replica lock lost". Keep
 * `work` to database operations: it is serialized with the other credential work of the lock.
 */
export const withLockConnection = <T>(
  scope: string,
  work: () => Promise<T>,
  { commit = false }: { commit?: boolean } = {},
): Promise<T> => {
  const owner = heldOwner(scope);
  if (!owner || owner.state === 'released') return work();
  if (owner.state === 'lost') return Promise.reject(lockLostError());
  const key = lockConnectionKey(scope);
  const enclosing = lockWork.getStore()?.get(key);
  if (enclosing?.active) {
    // Nested work joins the enclosing work, after the pieces that joined it before: concurrent
    // savepoints of one connection would release or roll back each other. The outermost work
    // commits.
    enclosing.root.commit ||= commit;
    const run = enclosing.queue
      .catch(() => undefined)
      .then(() => inSavepoint(owner, key, enclosing.root, work));
    enclosing.queue = run;
    return run;
  }
  // Not reentrant: an inherited queue marker would let a late task skip the queue.
  return withKeyedLock(
    key,
    async () => {
      // The lock may have been released while this work waited for its turn.
      if (owner.state === 'released') return work();
      const root: LockWork = { commit };
      const result = await inSavepoint(owner, key, root, work);
      if (root.commit) await commitAndAcquire(owner, scope);
      return result;
    },
    { reentrant: false },
  );
};

/**
 * Commit the lock transaction and take the lock again. A failed commit is thrown: the work is
 * not durable. A failed acquisition is not: the work is committed, so its caller goes on with
 * its result, and the lock is marked lost for what comes next.
 */
const commitAndAcquire = async (owner: LockOwner, scope: string): Promise<void> => {
  try {
    await owner.runner.commitTransaction();
  } catch (error) {
    owner.state = 'lost';
    throw error;
  }
  try {
    await acquire(owner.runner, scope);
  } catch (error) {
    owner.state = 'lost';
    logger.warn(`Replica lock lost after committing credential work: ${scope}`, error);
  }
};

/**
 * Run `work` on the lock connection in a savepoint, rolled back if `work` fails. Nested work
 * joining it runs inside the savepoint, even if `work` returns without waiting for it.
 */
const inSavepoint = async <T>(
  owner: LockOwner,
  key: string,
  root: LockWork,
  work: () => Promise<T>,
): Promise<T> => {
  // An earlier piece of work may have lost the lock.
  if (owner.state !== 'live') throw lockLostError();
  const { runner } = owner;
  const lost = () => (owner.state = 'lost');
  try {
    await runner.startTransaction(); // SAVEPOINT
  } catch (error) {
    lost();
    throw error;
  }
  const frame: LockWorkFrame = { root, active: true, queue: Promise.resolve() };
  let outcome: { result: T } | { error: unknown };
  try {
    outcome = {
      result: await lockWork.run(new Map(lockWork.getStore()).set(key, frame), () =>
        databaseTransaction.run(runner, work),
      ),
    };
  } catch (error) {
    outcome = { error };
  }
  frame.active = false;
  await frame.queue.catch(() => undefined);
  if ('error' in outcome) return rollbackAfter(runner, outcome.error, lost); // ROLLBACK TO SAVEPOINT
  try {
    await runner.commitTransaction(); // RELEASE SAVEPOINT
  } catch (error) {
    lost();
    throw error;
  }
  return outcome.result;
};

/**
 * In-process, per-key promise-chain lock, reentrant unless `reentrant: false`. `withReplicaLock` above
 * is a no-op outside replica mode because there is no cross-process lock to
 * take; this keeps same-key async work serialized within a single process
 * (e.g. a server's OAuth flows) in both modes.
 */
const heldKeyedLocks = new AsyncLocalStorage<Set<string>>();
const keyedQueues = new Map<string, Promise<unknown>>();

export const withKeyedLock = async <T>(
  key: string,
  action: () => Promise<T>,
  {
    timeoutMs,
    reentrant = true,
  }: {
    /** Give the turn up and reject after this delay. */
    timeoutMs?: number;
    /** Let nested calls of the same async context run at once; otherwise they queue too. */
    reentrant?: boolean;
  } = {},
): Promise<T> => {
  if (reentrant && heldKeyedLocks.getStore()?.has(key)) return action();

  const previous = keyedQueues.get(key) ?? Promise.resolve();
  let release!: () => void;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  const next = previous.catch(() => undefined).then(() => gate);
  keyedQueues.set(key, next);
  const turn = previous.catch(() => undefined);
  if (timeoutMs === undefined) {
    await turn;
  } else {
    let timer: NodeJS.Timeout | undefined;
    const timedOut = await Promise.race([
      turn.then(() => false),
      new Promise<boolean>((resolve) => {
        timer = setTimeout(() => resolve(true), timeoutMs);
      }),
    ]);
    clearTimeout(timer);
    if (timedOut) {
      // Give the turn up: the callers queued behind pass once the current holder is done.
      void turn.then(() => {
        release();
        if (keyedQueues.get(key) === next) keyedQueues.delete(key);
      });
      throw new Error('Timed out waiting for a lock held in this process; retry the request');
    }
  }

  // A non-reentrant holder leaves no marker for the tasks it starts to inherit.
  const held = reentrant ? new Set(heldKeyedLocks.getStore()).add(key) : undefined;
  try {
    return held ? await heldKeyedLocks.run(held, action) : await action();
  } finally {
    held?.delete(key);
    release();
    if (keyedQueues.get(key) === next) keyedQueues.delete(key);
  }
};

/**
 * Serialize one server's pending-OAuth writes (state + PKCE verifier) and the
 * callback's token exchange within this process, keyed by server name and
 * credential scope. Concurrent `connect()` calls (e.g. a double
 * enable/disable toggle) otherwise interleave on `pendingAuthorization`, and
 * a callback can exchange a code against a verifier stored for the wrong
 * state (`invalid_grant: Invalid PKCE code_verifier`).
 */
export const withOAuthPendingAuthLock = async <T>(
  serverName: string,
  username: string,
  action: () => Promise<T>,
): Promise<T> =>
  withReplicaLock(JSON.stringify(['oauth', serverName, username]), () =>
    withKeyedLock(JSON.stringify(['oauth-flow', serverName, username]), action),
  );
