# world

ETCDCTL_API=3 /home/qboxserver/ava-etcd/_package/etcdctl snapshot restore etcd_snapshot2.db \
  --name infra2 \
  --initial-cluster infra2=http://10.200.20.87:2380,infra2=http://10.200.20.89:2380,infra1=http://10.200.20.90:2380 \
  --initial-cluster-token etcd-cluster-1 \
  --initial-advertise-peer-urls http://10.200.20.87:2380


 ETCDCTL_API=3 /home/qboxserver/ava-etcd/_package/etcd --name infra1 --data-dir /disk1/ava-etcd --initial-advertise-peer-urls http://10.200.20.89:2380 \
  --listen-peer-urls http://10.200.20.89:2380 \
  --listen-client-urls http://10.200.20.89:2379,http://127.0.0.1:2379 \
  --advertise-client-urls http://10.200.20.89:2379 \
  --initial-cluster-token etcd-cluster-1 \
  --initial-cluster infra0=http://10.200.20.87:2380,infra1=http://10.200.20.89:2380,infra2=http://10.200.20.90:2380 \
  --initial-cluster-state new