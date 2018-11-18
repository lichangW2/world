RT=$(pwd)
PREFIX="http://10.200.30.13:10000/ozp6ez3lb.com0.z0.glb.clouddn.com"
RESULT="models_recover.txt"
BUCKET="atserving-model-privatebucket-z0"
rm $RESULT

for fe in $(ls)
do
   [ -f $RT/$fe/app.yaml ] || { continue; }
   lines=$(grep -E "tar_file|features\.line|labels\.line|extra_symbol_1|extra_model_1|label_cn|taglist_file" $RT/$fe/app.yaml)
    for ln in $lines
    do
              
             [ -z "lines" ] && { continue; }
             [[  ! $ln =~ ^\'http://.*|^http://.*  ]] && { continue; }
             ret=$(echo $ln | gsed -E "s/^.*glb\.clouddn\.com\/|^.*ozp6ez3lb\.bkt\.clouddn\.com\///")
             [ -z "${ret}" ] && { continue; }

             ret=${ret//\'/}
             cfile=${ret//\//-}
             echo "cfile : $cfile"
             qrsctl -v stat $BUCKET $ret 
             [ $? -ne 0 ] && { qrsctl -v mv $BUCKET:$cfile $BUCKET:$ret; } 
             [ $? -ne 1 ] && { echo "rename failed"; continue; } || { echo "modify deploy ..."; } #bugs
             gsed -i -E "s/http:.*glb\.clouddn\.com|http:.*ozp6ez3lb\.bkt\.clouddn\.com/http:\/\/10.200.30.13:10000\/pi6j8qepj.bkt.clouddn.com/" $RT/$fe/app.yaml
             #[ ! -z "${ret}" ] && { echo "${ret}" >> $RESULT; }
     done
done