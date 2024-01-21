## Real Time dat processing


## KAFKA for real time messagge transport and Queue

See these instructions to instlal KAFKA and run kafka service

https://kafka.apache.org/quickstart

## Test KAFKA publishing and subscriber

bin/kafka-console-producer.sh --topic [Your_Topic_Name] --bootstrap-server localhost:9092

bin/kafka-console-consumer.sh --topic [Your_Topic_Name] --from-beginning --bootstrap-server localhost:9092


## Train Model

Use spark_training.py to train the model


## Ingerence 

Use spark_inference.py to use trained model to infer on the streaming data coming from kafka message



