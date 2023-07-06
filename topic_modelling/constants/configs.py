import os

KAFKA_BOOTSTRAP_SERVERS = os.environ.get('KAFKA_BOOTSTRAP_SERVERS',"172.20.28.1:31390")
KAFKA_TOPICS = os.environ.get('KAFKA_TOPICS',"hermes02-preprocessing-entry")
KAFKA_GROUP_ID = os.environ.get('KAFKA_GROUP_ID',"hermes-output-post-images-01")
KAFKA_OFFSET_COMMIT = os.environ.get('KAFKA_OFFSET_COMMIT',"earliest")


if KAFKA_GROUP_ID is None:
    kafka_consumer_config = {
        "topic": KAFKA_TOPICS, # kafka topic
        "bootstrap_servers": KAFKA_BOOTSTRAP_SERVERS, # kafka bootstrap_servers
        "auto_offset_reset": KAFKA_OFFSET_COMMIT # data from beginning of topic
    }
else:
    kafka_consumer_config = {
    "topic": KAFKA_TOPICS, # kafka topic
    "bootstrap_servers": KAFKA_BOOTSTRAP_SERVERS, # kafka bootstrap_servers
    "group_id": KAFKA_GROUP_ID, # kafka group
    "auto_offset_reset": KAFKA_OFFSET_COMMIT # data from beginning of topic
    }




# MILVUS = {
#     "DATABASES": {
#         "default": {
#             "HOST": os.environ.get('MILVUS_HOST',"localhost"),
#             "PORT": os.environ.get('MILVUS_PORT',19530),
#         }
#     }
# }



MILVUS = {
    "DATABASES": {
        "default": {
            "HOST": os.environ.get('MILVUS_HOST',"172.20.28.1"),
            "PORT": os.environ.get('MILVUS_PORT',32602),
        }
    }
}


milvus_config = {"alias": 'default', # a collection created without specifying the consistency level is set with Bounded consistency level
                "host": MILVUS['DATABASES']['default']['HOST'], 
                "port": MILVUS['DATABASES']['default']['PORT']
                }


               