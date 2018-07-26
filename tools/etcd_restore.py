import os
import base64
import json


etcd_file=open("etcd_data.txt","r")
etcd_data=json.load(etcd_file)

for data in etcd_data["kvs"]:
    key,value=data["key"],data["value"]
    key=base64.standard_b64decode(key)
    value=base64.standard_b64decode(value)
    if value.startswith("-"):
        os.system(
            "ETCDCTL_API=3 /home/qboxserver/ava-etcd/_package/etcdctl  --endpoints=http://10.200.20.87:2379,http://10.200.20.89:2379,http://10.200.20.90:2379  put " + key + " -- '" + value + "'")
        continue

    os.system("ETCDCTL_API=3 /home/qboxserver/ava-etcd/_package/etcdctl  --endpoints=http://10.200.20.87:2379,http://10.200.20.89:2379,http://10.200.20.90:2379  put "+key+" '"+value+"'")



root="/Users/cj/qiniu/deploy-test/playbook/ava-serving/apps"
apps=($(ls apps | xargs))
for app in $apps; do sed '' -i 's/http:\/\/10.200.30.13:2379,http:\/\/10.200.30.14:2379,http:\/\/10.200.30.15:2379/http:\/\/10.200.20.87:2379,http:\/\/10.200.20.89:2379,http:\/\/10.200.20.90:2379/' $root/$app/app.yaml; done