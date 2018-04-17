import aerospike
import time
from apscheduler.schedulers.background import BackgroundScheduler
import random
import logging
import Logger
import Aerospike_Connect


# Scheduler
sched = BackgroundScheduler()

@sched.scheduled_job('cron', second='0-59') # kviecia kai tik prasideda sekunde
def scheduled_job():
    loadData(int(time.time()), randomData())

# Configure the client
config = {
  'hosts': [ ('172.28.128.3', 3000) ]
}

# Create a client and connect it to the cluster
try:
  client = aerospike.client(config).connect()
except:
  import sys
  print("failed to connect to the cluster with", config['hosts'])
  sys.exit(1)
# Data loading into aerospike server function
def loadData(key, data):
    print("Writting: ", key, " : ", data)
    try:
        client.put(('test', 'HRPI', key), { 'value': data })
    except Exception as e:
        import sys
        print("error: {0}".format(e), file=sys.stderr)
# Random int generator
def randomData():
    return random.randint(0, 100)

# start scheduler
sched.start()
input() #Press ENTER to continue
# stop scheduler
sched.shutdown()
client.close()


Logger = Logger.Logger_Class()
logger = logging.getLogger(__name__)

logger.info('SourceModule')

dbclient = Aerospike_Connect.AerospikeClient()
dbclient.connect()
