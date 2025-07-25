# Gunicorn configuration file
import multiprocessing

max_requests = 1000
max_requests_jitter = 50

log_file = "/var/log/lizardmorph"

# Log file configurations
accesslog = "/var/log/lizardmorph/access.log"
errorlog = "/var/log/lizardmorph/error.log"

# Daemon configuration
daemon = True

bind = "0.0.0.0:3000"

workers = (multiprocessing.cpu_count() * 2) + 1
threads = workers

timeout = 120