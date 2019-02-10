import os
import sys
import time
import getopt
import json
import random
import hashlib
import string

import traceback
import multiprocessing as mp


from kafka import KafkaConsumer
from kafka import KafkaProducer

LOG='{"time":%d,"uid":"%d","req":{"uri":"%s"}}'

HELP="""
kafka_consumer_publisher [options]
  [options]:
    -h[--help]: help info
    -p[--producer]: start as the info producers 
    -c[--consumer]: start as the consumers
"""

def KFProducer(proc_id,topic,server):
    kp=KafkaProducer(bootstrap_servers=server, value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                  compression_type='gzip')
    sys.stdout.flush()
    try:
        while True:
            print("here:",proc_id)
            time.sleep(1)
            kp.send(topic,LOG%(int(time.time()),proc_id,''.join(random.sample(string.ascii_letters+string.digits,10))))
    except Exception as _e:
        print("error: ",type(_e),_e)
        print(traceback.format_exc())


def KFConsumer(worker,topic, server):
    sys.stdout.flush()
    print("start consumer... ",str(worker)+"test_gp")
    consumer= KafkaConsumer(topic,group_id = str(worker)+"test_gp", bootstrap_servers=server,value_deserializer=lambda v: json.loads(v) )
    for msg in consumer:
        time.sleep(1)
        print("worker:{}, receive one message: {}\n".format(worker,msg))


if __name__=="__main__":

    opts,_=getopt.getopt(sys.argv[1:],"-h-p-c",["help","producer","consumer"])
    topic="test-kafka"
    server=['localhost:9092']

    try:
        for opt_name, opt_args in opts:
            print("opt_name %s, opt_args %s"%(opt_name, opt_args))
            if opt_name in ("-h","--help"):
                print(HELP)
            elif opt_name in ("-p","--producer"):
                # producer is processing safe
                pool=mp.Pool(processes=3)
                for id in xrange(3):
                    print("id ",id)
                    ret=pool.apply_async(KFProducer,args=(id,topic,server,))

                print("3 producers is running...")
                pool.close()
                pool.join()
            elif opt_name in ("-c","--consumer"):
                pool = mp.Pool(processes=2)
                for id in xrange(2):
                    ret = pool.apply_async(KFConsumer, args=(id, topic,server,))

                print("2 consumers is running...")
                pool.close()
                pool.join()
            else:
                print(HELP)
            sys.exit(0)
        print(HELP)
    except Exception as _e:
        print(traceback.format_exc())
