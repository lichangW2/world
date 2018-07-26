#!/usr/bin/env bash

function deployApp(){
    cmd=$1
    root_path=$2
    appss=($@)
    apps=${appss[@]:2}
    echo "cmd: "$cmd"  apps: "$apps"  root_path: "$root_path

    for app in $apps
    do
       if [ -d $root_path/$app ]
          then
            info=$(curl -v  -H "Authorization:QiniuStub uid=1381102897&ut=1" http://10.200.20.54:8001/v1/ufops/$app/info/brief)
            image=$(python -c "print($info['releases'][0]['image']).split('/')[1]")
            if [ $? -eq 0 ];then
                  echo "start deploy app: "$app" with image: " $image
                  $cmd deploy $app -i $image  -d "cs machine migration"
                  if [ $? -ne 0 ];then
                        echo "deploy failed app " $app >> $failed
                  fi
            else
               echo "deploy failed app " $app >> $failed
            fi
       fi
    done
}



root_ava_serving=/root/opt/fdeploy/playbook/ava-serving/apps
root_ava_argus=/root/opt/fdeploy/playbook/ava-argus/apps
ava_serving=$(ls $root_ava_serving)
ava_argus=$(ls $root_ava_argus)
#apps=$(curl -v  -H "Authorization:QiniuStub uid=1381102897&ut=1" http://10.200.20.54:8001/v1/ufops)
#apps=("${ava_serving[@]}" "${ava_argus[@]}")

failed="deploy_failed.txt"

deployApp "ava"  $root_ava_serving $ava_serving
deployApp "argus" $root_ava_argus $ava_argus
