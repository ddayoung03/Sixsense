"""Quick GPU sanity check.  source env.sh first."""
import time
import tensorflow as tf

print("TF", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus)
if not gpus:
    raise SystemExit("no GPU visible - check env.sh / nvidia-smi in WSL")

a = tf.random.normal((2048, 2048))
b = tf.random.normal((2048, 2048))
(a @ b).numpy()                       # warm up (first Blackwell call JIT-compiles)
t = time.time()
for _ in range(20):
    c = a @ b
c.numpy()
print("matmul 20x 2048^3: %.3fs   device=%s" % (time.time() - t, c.device))
