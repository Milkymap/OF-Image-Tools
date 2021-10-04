import cv2 
import zmq 
import json 
import numpy as np 

from libraries.log import logger 
from libraries.strategies import * 

from tqdm import tqdm
from os import getenv, path 

class ZMQImageMatcher:
    def __init__(self, source, target, server_port, threshold, shape):
        self.server_port = server_port 
        self.shape = shape 
        self.source = source 
        self.target = target 
        self.threshold = threshold 
        self.matcher = cv2.BFMatcher()
        self.extractor = cv2.SIFT_create()
    
    def get_sift(self, image):
        keypoints, descriptor = self.extractor.detectAndCompute(image, None) 
        return keypoints, descriptor

    def turn_sift2mbrsift(self, descriptor):
        sink = []
        for row in descriptor:
            breaked_row = np.split(row, 4)
            breaked_row.reverse()
            chunks_acc = []
            for groups in breaked_row:
                breaked_chunk = np.split(groups, 4)
                for chunk in breaked_chunk:
                    head, *remainder = chunk
                    new_chunk = np.hstack([head, np.flip(remainder)])
                    chunks_acc.append(new_chunk)
                # end loop chunk ...!
            # end loop group ...!
            sink.append(row)
            sink.append(np.hstack(chunks_acc))
        # end loop row ...!
        return np.vstack(sink) 

    def compare_descriptors(self, source_des, target_des):
        matches = self.matcher.knnMatch(source_des, target_des, k=2)
        valid_matches = 0 
        for left, right in matches:
            if left.distance < self.threshold * right.distance:
                valid_matches = valid_matches + 1 
        return valid_matches

    def start(self):
        INITIALIZED = False  
        try:
            ctx = zmq.Context()
            router_socket = ctx.socket(zmq.ROUTER)
            router_socket.bind(f'tcp://*:{self.server_port}')
            router_controller = zmq.Poller()
            router_controller.register(router_socket, zmq.POLLIN)

            keep_routing = True 
            while keep_routing:
                logger.debug(f'server is listening on port [{self.server_port}]')
                incoming_events = dict(router_controller.poll(100))
                if router_socket in incoming_events:
                    if incoming_events[router_socket] == zmq.POLLIN: 
                        client_address, _, client_message = router_socket.recv_multipart()
                        try:
                            loaded_message = json.loads(client_message.decode())
                            
                            path2candidate = path.join(self.source, loaded_message['candidate'])
                            path2neighbors = [ path.join(self.target, elm) for elm in loaded_message['neighbors']]

                            src_image = read_image(path2candidate, size=self.shape)
                            src_keypoints, src_descriptor = self.get_sift(src_image)
                            src_descriptor = self.turn_sift2mbrsift(src_descriptor)

                            retained = []
                            for target in tqdm(path2neighbors):
                                trg_image = read_image(target, size=self.shape)
                                trg_keypoints, trg_descriptor = self.get_sift(trg_image)
                                trg_descriptor = self.turn_sift2mbrsift(trg_descriptor)
                                
                                matching_weight = self.compare_descriptors(src_descriptor, trg_descriptor)
                                matching_weight /= (2 * np.maximum(len(src_keypoints), len(trg_keypoints)))
                                
                                if matching_weight > 0.1:
                                    _, target_name = path.split(target)
                                    retained.append({'image_id': target_name, 'score': matching_weight})
                            
                            response = json.dumps({
                                'global_status': 1,
                                'error_message': '', 
                                'response': {
                                    'local_status':  len(retained) > 0, 
                                    'duplicated': retained
                                }
                            }).encode()

                        except Exception as e:
                            logger.error(f'an error occurs during instance matching {e}')
                            response = json.dumps({
                                'global_status':  0,
                                'error_message': f'[{e}]', 
                                'response': {}
                            }).encode()

                        router_socket.send_multipart([client_address, b'', response])
                        logger.success(f'server finishs to search duplication for incoming request {loaded_message["candidate"]}')
            # end loop 
        except KeyboardInterrupt as e:
            pass 
        except Exception as e:
            logger.error(e)
        finally:
            if INITIALIZED:
                router_controller.unregister(router_socket)
                router_socket.close()
                ctx.term() 

def main():
    SOURCE = getenv('SOURCE')
    TARGET = getenv('TARGET')
    SERVER_PORT = int(getenv('SERVER_PORT'))
    THRESHOLD= float(getenv('THRESHOLD'))
    WIDTH= int(getenv('WIDTH'))
    HEIGHT = int(getenv('HEIGHT'))

    server = ZMQImageMatcher(SOURCE, TARGET, SERVER_PORT, THRESHOLD, (WIDTH, HEIGHT))
    server.start()

if __name__ == '__main__':
    main()