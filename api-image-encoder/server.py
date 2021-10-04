import zmq 
import json 

import torch as th 
import torch.nn as nn 

from os import path, getenv 
from libraries.log import logger 
from libraries.strategies import * 

class ZMQVectorizer:
    def __init__(self, server_port, models_path, models_name, images_path):
        self.server_port = server_port 
        self.models_path = models_path 
        self.images_path = images_path 
        self.models_name = models_name 
        
        self.device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
        self.vectorizer = self.__load_model(models_path, models_name, self.device)
    
    def __load_model(self, path2models, model_name, target_device):
        model = th.load(path.join(path2models, model_name), map_location=target_device)
        specialized_model = nn.Sequential(*list(model.children())[:-1]).eval()
        
        for prm in specialized_model.parameters():
            prm.requires_grad = False 

        return specialized_model
        
    def start_service(self):
        ZMQ_INITIALIZED = False 
        try:
            context = zmq.Context()
            router_socket = context.socket(zmq.ROUTER)
            router_socket.bind(f'tcp://*:{self.server_port}')
            router_controller = zmq.Poller()
            router_controller.register(router_socket, zmq.POLLIN)
            ZMQ_INITIALIZED = True 

            keep_routing = True 
            while keep_routing:
                logger.debug(f'server is listening on port {self.server_port}')
                incoming_events = dict(router_controller.poll(100))
                if router_socket in incoming_events:
                    if incoming_events[router_socket] == zmq.POLLIN: 
                        remote_address, delimeter, message = router_socket.recv_multipart()
                        try:
                            json_contents = json.loads(message.decode())
                            image_name = json_contents['image_name']
                            path2image = path.join(self.images_path, image_name)

                            if path.isfile(path2image):
                                try:
                                    bgr_image = read_image(path2image)
                                    tensor_3d = cv2th(bgr_image)
                                    prepared_tensor_3d = prepare_image(tensor_3d)
                                    fingerprint = th.squeeze(
                                        self.vectorizer(prepared_tensor_3d[None, ...])
                                    ).numpy().tolist()
                                    status = 1 
                                except Exception as e:
                                    logger.error('an error occur during image (reading | vectorization) process :', e)
                                    fingerprint = []
                                    status = 0
                            else:
                                logger.error(f'{path2image} is not a valid path')
                                fingerprint = []
                                status = 0 
                        except Exception as e:
                            logger.error(f'an error {e} occurs during json or path handler ...!')
                    
                        response = json.dumps({
                            'status': status, 
                            'fingerprint': fingerprint 
                        }).encode()

                        router_socket.send_multipart([remote_address, delimeter, response])
                        logger.success(f'server responded to {remote_address}')
                    # end if incoming pollin events  
                # end if incoming events 
            # end loop routing 

        except KeyboardInterrupt as e:
            logger.error(e)
        except Exception as e: 
            logger.error(e)
        finally:
            if ZMQ_INITIALIZED:
                router_controller.unregister(router_socket)
                router_socket.close()
                context.term()
                logger.success('all zmq ressources were freed')

def main():
    MODELS_PATH = getenv('MODELS_PATH')
    IMAGES_PATH = getenv('IMAGES_PATH')
    SERVER_PORT = getenv('SERVER_PORT')
    MODELS_NAME = getenv('MODELS_NAME')

    server = ZMQVectorizer(SERVER_PORT, MODELS_PATH, MODELS_NAME, IMAGES_PATH)
    server.start_service()


if __name__ == '__main__':
    main()

