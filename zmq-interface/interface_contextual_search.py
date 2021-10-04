import zmq 
import click 
import numpy as np
import json 

import time 

from os import path 
from loguru import logger 
from glob import glob 

@click.command()
@click.option('--server_address', help='address of the remote app', required=True)
def interface(server_address):
    try:
        ctx = zmq.Context()
        
        dealer_socket = ctx.socket(zmq.DEALER)
        dealer_socket.setsockopt_string(zmq.IDENTITY, '300')
        dealer_socket.connect(server_address)
        dealer_controller = zmq.Poller()
        dealer_controller.register(dealer_socket, zmq.POLLIN)

        keep_interface = True 
        while keep_interface:
            incoming_events = dict(dealer_controller.poll(100))
            if dealer_socket in incoming_events:
                if incoming_events[dealer_socket] == zmq.POLLIN:
                    _, contents = dealer_socket.recv_multipart()
                    decoded_contents = json.loads(contents.decode())
                    if decoded_contents['status'] == 1:
                        response = decoded_contents['neighbors']
                        print(response)
                    else:
                        print('The server was not able to handle this request')
        
            fingerprint = np.random.normal(size=512)
            request2send = json.dumps({
                'nb_neighbors': 16,
                'vec_features': fingerprint.tolist()
            }).encode()

            dealer_socket.send_multipart([b'', request2send])
            time.sleep(0.1)

        # end loop 
    except KeyboardInterrupt as e:
        pass 
    except Exception as e:
        logger.error(e)
    finally:
        dealer_controller.unregister(dealer_socket)
        dealer_socket.disconnect(server_address)
        dealer_socket.close()

if __name__ == '__main__':
    interface()