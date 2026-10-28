CFLAGS += -Wall

seccomp-notify: seccomp-notify.c

run:
	./run.sh

clean:
	rm -rf seccomp-notify dirs notify.sock child.pid child.log supervisor.log *.img dump.log restore.log stats-dump stats-restore

.PHONY: run clean
