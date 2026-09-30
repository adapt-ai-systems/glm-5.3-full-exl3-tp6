"""stdin -> stdout at most RATE MB/s (default 50). pv stand-in."""
import sys, time
rate = float(sys.argv[1]) * 1e6 if len(sys.argv) > 1 else 50e6
i, o, t0, n = sys.stdin.buffer, sys.stdout.buffer, time.time(), 0
while b := i.read(1 << 20):
    o.write(b); n += len(b)
    ahead = n / rate - (time.time() - t0)
    if ahead > 0:
        time.sleep(ahead)
o.flush()
