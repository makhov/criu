/*
 * Supervisor and child for the --seccomp-notify-socket test.
 *
 * The child installs a filter that sends mkdirat() to a listener and then
 * creates dirs/N in a loop, logging each result. The supervisor answers
 * every notification with SECCOMP_USER_NOTIF_FLAG_CONTINUE. Listeners for
 * restored tasks arrive on the SOCK_SEQPACKET socket "notify.sock".
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <poll.h>
#include <signal.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <unistd.h>

#define MAX_FDS 64

static void send_fd(int sk, int fd)
{
	char c = 0, buf[CMSG_SPACE(sizeof(int))] = {};
	struct iovec iov = { .iov_base = &c, .iov_len = 1 };
	struct msghdr h = { .msg_iov = &iov, .msg_iovlen = 1, .msg_control = buf, .msg_controllen = sizeof(buf) };
	struct cmsghdr *ch = CMSG_FIRSTHDR(&h);

	ch->cmsg_level = SOL_SOCKET;
	ch->cmsg_type = SCM_RIGHTS;
	ch->cmsg_len = CMSG_LEN(sizeof(int));
	memcpy(CMSG_DATA(ch), &fd, sizeof(int));
	if (sendmsg(sk, &h, 0) < 0) {
		perror("sendmsg");
		exit(1);
	}
}

/* Returns the fd and, if tid is set, the int32 payload criu sends with it. */
static int recv_fd(int sk, int *tid)
{
	char buf[CMSG_SPACE(sizeof(int))] = {};
	int payload = 0, fd;
	struct iovec iov = { .iov_base = &payload, .iov_len = sizeof(payload) };
	struct msghdr h = { .msg_iov = &iov, .msg_iovlen = 1, .msg_control = buf, .msg_controllen = sizeof(buf) };
	struct cmsghdr *ch;

	if (recvmsg(sk, &h, 0) <= 0)
		return -1;
	ch = CMSG_FIRSTHDR(&h);
	if (!ch || ch->cmsg_type != SCM_RIGHTS)
		return -1;
	memcpy(&fd, CMSG_DATA(ch), sizeof(int));
	if (tid)
		*tid = payload;
	return fd;
}

static void child(int sk)
{
	struct sock_filter f[] = {
		BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, nr)),
		BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_mkdirat, 0, 1),
		BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_USER_NOTIF),
		BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
	};
	struct sock_fprog prog = { .len = sizeof(f) / sizeof(f[0]), .filter = f };
	FILE *log;
	int lfd, fd;
	long i;

	if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)) {
		perror("PR_SET_NO_NEW_PRIVS");
		exit(1);
	}
	lfd = syscall(__NR_seccomp, SECCOMP_SET_MODE_FILTER, SECCOMP_FILTER_FLAG_NEW_LISTENER, &prog);
	if (lfd < 0) {
		perror("seccomp");
		exit(1);
	}
	send_fd(sk, lfd);
	close(lfd);
	close(sk);

	setsid();
	fd = open("/dev/null", O_RDWR);
	dup2(fd, 0);
	dup2(fd, 1);
	dup2(fd, 2);
	close(fd);

	log = fopen("child.log", "a");
	setvbuf(log, NULL, _IOLBF, 0);
	for (i = 1;; i++) {
		char d[64];
		int r;

		snprintf(d, sizeof(d), "dirs/%ld", i);
		r = mkdir(d, 0755);
		fprintf(log, "%ld %d %d\n", i, r, r ? errno : 0);
		usleep(100000);
	}
}

/* Answers one notification; returns -1 once the listener's tasks are gone. */
static int serve(int lfd)
{
	struct seccomp_notif_sizes sizes;
	struct seccomp_notif *req;
	struct seccomp_notif_resp *resp;
	int ret = -1;

	if (syscall(__NR_seccomp, SECCOMP_GET_NOTIF_SIZES, 0, &sizes))
		return -1;
	req = calloc(1, sizes.seccomp_notif);
	resp = calloc(1, sizes.seccomp_notif_resp);
	if (!req || !resp)
		goto out;
	if (ioctl(lfd, SECCOMP_IOCTL_NOTIF_RECV, req) < 0) {
		ret = errno == EINTR ? 0 : -1;
		goto out;
	}
	resp->id = req->id;
	resp->flags = SECCOMP_USER_NOTIF_FLAG_CONTINUE;
	/* ENOENT: the task was killed (e.g. by criu dump) while notified. */
	if (ioctl(lfd, SECCOMP_IOCTL_NOTIF_SEND, resp) < 0 && errno != ENOENT)
		perror("SECCOMP_IOCTL_NOTIF_SEND");
	ret = 0;
out:
	free(req);
	free(resp);
	return ret;
}

int main(void)
{
	struct sockaddr_un addr = { .sun_family = AF_UNIX, .sun_path = "notify.sock" };
	struct pollfd fds[MAX_FDS] = {};
	int kind[MAX_FDS] = {}; /* 0: notify socket, 1: connection, 2: listener */
	int lsk, sv[2], n = 0, i;
	FILE *pidf;
	pid_t pid;

	/* Reap the child when criu dump kills it, so restore can reuse its pid. */
	signal(SIGCHLD, SIG_IGN);
	setvbuf(stdout, NULL, _IOLBF, 0);

	lsk = socket(AF_UNIX, SOCK_SEQPACKET, 0);
	unlink(addr.sun_path);
	if (lsk < 0 || bind(lsk, (struct sockaddr *)&addr, sizeof(addr)) || listen(lsk, 8)) {
		perror("notify socket");
		return 1;
	}
	if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv)) {
		perror("socketpair");
		return 1;
	}

	pid = fork();
	if (pid < 0) {
		perror("fork");
		return 1;
	}
	if (pid == 0) {
		close(sv[0]);
		close(lsk);
		child(sv[1]);
	}
	close(sv[1]);

	fds[n].fd = lsk;
	fds[n].events = POLLIN;
	kind[n++] = 0;
	fds[n].fd = recv_fd(sv[0], NULL);
	fds[n].events = POLLIN;
	kind[n++] = 2;
	close(sv[0]);
	if (fds[1].fd < 0) {
		fprintf(stderr, "no listener from the child\n");
		return 1;
	}

	pidf = fopen("child.pid", "w");
	fprintf(pidf, "%d\n", pid);
	fclose(pidf);

	for (;;) {
		if (poll(fds, n, -1) < 0)
			continue;
		for (i = 0; i < n; i++) {
			int tid, l;

			if (!fds[i].revents)
				continue;
			switch (kind[i]) {
			case 0:
				if (n == MAX_FDS)
					break;
				fds[n].fd = accept(lsk, NULL, NULL);
				fds[n].events = POLLIN;
				kind[n++] = 1;
				break;
			case 1:
				l = recv_fd(fds[i].fd, &tid);
				if (l < 0 || n == MAX_FDS) {
					close(fds[i].fd);
					fds[i].fd = -1;
					break;
				}
				printf("restored listener for tid %d\n", tid);
				fds[n].fd = l;
				fds[n].events = POLLIN;
				kind[n++] = 2;
				break;
			case 2:
				if ((fds[i].revents & POLLIN) && serve(fds[i].fd) == 0)
					break;
				close(fds[i].fd);
				fds[i].fd = -1;
				break;
			}
		}
	}
}
