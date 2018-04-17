import logging
import aerospike

class AerospikeClient:

    # Configure the client
    def __init__(self, ip_address='172.28.128.3', port=3000, namespace='test', db_set='HRPI_one'):
        self.config = { 'hosts': [ (ip_address, port)] }
        self.ip_address = ip_address
        self.port = port
        self.namespace = namespace
        self.db_set = db_set
        self.logger = logging.getLogger(__name__)

    # Create a client and connect it to the cluster
    def connect(self):
        try:
            self.client = aerospike.client(self.config).connect()
        except Exception as e:
            import sys
            self.logger.error('Failed to connect to the cluster with ' + str(self.config['hosts']) + 'Exception message: ' + str(e))

    def read_value(self, key):
        if self.client.exists(self.namespace, self.db_set, key)[1] is not None:
            return self.client.get((self.namespace, self.db_set, key))[2].get('HRPI')
        else:
            return -1

    def read_value_range(self, start_range, end_range):
        try:
            values = []
            for key in range(start_range, end_range):
                values.append(self.read_value(key))
        except Exception as e:
            self.logger.error('Reading value failed with host: ' + str(self.config['hosts']) + ', KEY=' + str(key) + 'Exception message: ' + str(e))

    def put_data(self, hrpi_time, hrpi):
        bins = {
                'HRPI': hrpi
               }
        key = (self.namespace, self.db_set, hrpi_time)
        try:
            self.client.put(key, bins)
        except Exception as e:
            self.logger.error('Inserting value failed with host: ' + str(self.config['hosts']) + ', KEY=' + str(key) + 'Exception message: ' + str(e))