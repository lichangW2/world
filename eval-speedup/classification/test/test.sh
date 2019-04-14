#!/bin/bash
echo `date`
for i in {1..100}
do
curl -XPOST --data-binary @/workspace/serving/models/image/bomb.jpg localhost:8000/api/classify 1>/dev/null 2>/dev/null
curl -XPOST --data-binary @/workspace/serving/models/image/sexy2.jpg localhost:8000/api/classify 1>/dev/null 2>/dev/null
curl -XPOST --data-binary @/workspace/serving/models/image/po_xjp.jpeg localhost:8000/api/classify 1>/dev/null 2>/dev/null
curl -XPOST --data-binary @/workspace/serving/models/image/pulp2 localhost:8000/api/classify 1>/dev/null 2>/dev/null
done
echo `date`
