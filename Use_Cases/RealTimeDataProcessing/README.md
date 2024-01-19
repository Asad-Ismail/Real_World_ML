## Real Time dat processing


## KAFKA for real time messagge transport and Queue

See these instructions to instlal KAFKA and run kafka service

https://kafka.apache.org/quickstart

## Test KAFKA publishing and subscriber

bin/kafka-console-producer.sh --topic [Your_Topic_Name] --bootstrap-server localhost:9092

bin/kafka-console-consumer.sh --topic [Your_Topic_Name] --from-beginning --bootstrap-server localhost:9092



