#!/usr/bin/env bash
# A task whose filter returns SECCOMP_RET_USER_NOTIF is dumped and restored
# with --seccomp-notify-socket. After restore, the supervisor must receive
# the new listener, the notified mkdir() calls must succeed again, and the
# restored task must not hold a listener itself.

set -o errexit
set -o nounset
set -o pipefail
set -o xtrace

source ../env.sh

make clean
make seccomp-notify
mkdir dirs

./seccomp-notify > supervisor.log 2>&1 &
supervisor=$!
trap 'kill $supervisor 2>/dev/null || true; [ -n "${child:-}" ] && kill -9 "$child" 2>/dev/null || true' EXIT

for _ in $(seq 50); do
	[ -s child.pid ] && [ -s child.log ] && break
	sleep 0.1
done
child=$(cat child.pid)

"$criu" dump --tree="$child" --shell-job --verbosity=4 --log-file=dump.log
before=$(wc -l < child.log)

"$criu" restore --restore-detached --shell-job --verbosity=4 --log-file=restore.log \
	--seccomp-notify-socket "$PWD/notify.sock"
sleep 1

if ! grep -q "restored listener for tid" supervisor.log; then
	echo "Supervisor did not receive the restored listener"
	echo FAIL
	exit 1
fi

# Every mkdir after the restore must have gone through the supervisor.
if tail -n +"$((before + 1))" child.log | awk '$2 != 0 { bad = 1 } END { exit !(NR > 0 && !bad) }'; then
	:
else
	echo "mkdir failed or did not run after restore:"
	tail -n +"$((before + 1))" child.log
	echo FAIL
	exit 1
fi

if ls -l "/proc/$child/fd" | grep -q "anon_inode:seccomp"; then
	echo "Restored task still holds a seccomp listener"
	echo FAIL
	exit 1
fi

echo PASS
