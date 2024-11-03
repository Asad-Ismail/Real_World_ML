kubectl get pods
cd src
kubectl create -f 1.1-basic_pod.yaml
kubectl get pods
kubectl describe pod mypod | more
kubectl delete pod mypod
kubectl create -f 1.2-port_pod.yaml
kubectl describe pod mypod | more
curl 192.168.###.###:80 (Replace ###.### with the IP address octets from the describe output)
# This command will time out (see the next lesson to understand why)
kubectl describe pod mypod | more
kubectl delete pod mypod
kubectl create -f 1.4-resources_pod.yaml
kubectl describe pod mypod | more