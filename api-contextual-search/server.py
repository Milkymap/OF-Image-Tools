import zmq 
import tqdm 
import json 

import ngtpy 

import numpy as np 
import pickle as pk 

import operator as op 
import itertools as it, functools as ft 

import torch as th 
import torch.nn as nn 
import multiprocessing as mp 

from os import mkdir, getenv
from libraries.log import logger 
from libraries.strategies import * 

from torch.utils.data import TensorDataset, DataLoader

class ZMQNGT:
    def __init__(self, batch_size, router_port, index, dimension, distance_type, features_location):
        self.index = index 
        self.dimension = dimension
        self.batch_size = batch_size 
        self.router_port = router_port
        self.distance_type = distance_type
        self.features_location = features_location
        
        self.ngt_location = path.join(index, 'ngt_dump')
        self.idx2img_id_location = path.join(index, 'idx2img_id.pkl')

        if not path.isdir(self.index):
            mkdir(self.index)
            ngtpy.create(self.ngt_location.encode(), dimension=self.dimension, distance_type=self.distance_type)
            self.engine = ngtpy.Index(self.ngt_location.encode())
            logger.debug('creation of the index ...!')
            vector_paths = sorted(pull_files(self.features_location, extension='*.pkl'))
            self.idx2img_id = []
            for v_path in vector_paths:
                logger.debug(f'indexation of the packet {v_path}')
                with open(v_path, 'rb') as fp:
                    loaded_data = pk.load(fp)
                    self.idx2img_id.extend(list(loaded_data.keys()))
                    tensor_data = th.as_tensor(np.vstack(list(loaded_data.values())))
                    dataset = TensorDataset(tensor_data)
                    dataloader = DataLoader(dataset, shuffle=False, batch_size=self.batch_size)

                    for tensor_batch in tqdm.tqdm(dataloader):
                        self.engine.batch_insert(th.vstack(tensor_batch).numpy())
                # end context manaer for pickle open 
            # end for loop over vector packet
             
            self.engine.save()
            with open(self.idx2img_id_location, 'wb') as fp:
                pk.dump(self.idx2img_id, fp)
            # next step add a status serializer in order to know if the index was created successfully or not
            # this flag will be interesting for next run
        else:
            self.engine = ngtpy.Index(self.ngt_location.encode())
            with open(self.idx2img_id_location, 'rb') as fp:
                self.idx2img_id = pk.load(fp) 
            
        # end if ...! 
         
    def start(self):
        ZMQ_INITIALIZED = False 
        try:
            ctx = zmq.Context() 
            router_socket = ctx.socket(zmq.ROUTER)
            router_socket.bind(f'tcp://*:{self.router_port}')
            router_controller = zmq.Poller()
            router_controller.register(router_socket, zmq.POLLIN)

            ZMQ_INITIALIZED = True 
            logger.success('router server is initialized 100%')

            keep_routing = True 
            while keep_routing:
                incoming_events = dict(router_controller.poll(100))
                logger.debug(f'server is listening on port {self.router_port}')

                if router_socket in incoming_events:
                    if incoming_events[router_socket] == zmq.POLLIN:
                        remote_address, _, remote_request = router_socket.recv_multipart()
                            
                        try:
                            decoded_request = json.loads(remote_request.decode())
                            nb_neighbors = decoded_request['nb_neighbors']
                            vec_features = decoded_request['vec_features']
                            ngt_response = self.engine.search(vec_features, nb_neighbors)
                            positions, distances = list(zip(*ngt_response))
                            selected_candidates = op.itemgetter(*positions)(self.idx2img_id)    
                            zipped_solutions = list(zip(selected_candidates, distances))
                            mapped_solutions = [ {'path': itm[0], 'score': itm[1]} for itm in zipped_solutions ]
                            response2send = json.dumps({
                                'global_status': 1, 
                                'error_message': '',
                                'response': {
                                    'local_status': 1, 
                                    'neighbors': mapped_solutions
                                }
                            }).encode()
                        except Exception as e:
                            logger.error(f'an error occurs during request handler {e}')
                            response2send = json.dumps({
                                'global_status': 1, 
                                'error_message': f'[{e}]',
                                'response': {}
                            }).encode()
                        
                        router_socket.send_multipart([remote_address, b'', response2send])
                        logger.success(f'the server send new response to client {remote_address.decode()}')
                    # end if zmq pollin events  
                # end if incoming events  
            # end routing loop 

        except KeyboardInterrupt as e:
            logger.warning(e)
        except Exception as e: 
            logger.error(e)
        finally:
            if ZMQ_INITIALIZED:
                router_controller.unregister(router_socket)
                router_socket.close()
                ctx.term()
            logger.success('zmq ressources free  100%')

def main():
    INDEX = getenv('INDEX')
    DIMENSION = int(getenv('DIMENSION'))
    BATCH_SIZE = int(getenv('BATCH_SIZE'))
    ROUTER_PORT = int(getenv('SERVER_PORT'))
    DISTANCE_TYPE = getenv('DISTANCE_TYPE')
    FEATURES_LOCATION = getenv('FEATURES_LOCATION')
    
    server = ZMQNGT(BATCH_SIZE, ROUTER_PORT, INDEX, DIMENSION, DISTANCE_TYPE, FEATURES_LOCATION)
    server.start()

if __name__ == '__main__':
    logger.debug(' ... [NGT server] ... ')
    main()
    