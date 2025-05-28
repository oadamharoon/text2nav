!/bin/bash
JUPYTER=$(which jupyter)
if [ -z "$JUPYTER" ]; then
echo " "
echo "ERROR: Jupyter was not found. Exiting."
echo " "
exit
fi
if [ -z "$JS2_PUBLIC_IP" ]; then
JS2_PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)
fi
#echo "JS2 IP is: $JS2_PUBLIC_IP"
$JUPYTER lab --no-browser --ip=0.0.0.0 2>&1 | sed s/127.0.0.1/${JS2_PUBLIC_IP}/g
