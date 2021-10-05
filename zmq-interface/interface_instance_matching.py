import zmq 
import click 
import numpy as np
import json 

import time 

from os import path 
from loguru import logger 
from glob import glob 

@click.command()
@click.option('--size', help='how many image to use for test', type=int)
@click.option('--source', help='path to source dataset')
@click.option('--server_address', help='address of the remote app', required=True)
def interface(size, source, server_address):
    try:
        ctx = zmq.Context()
        
        dealer_socket = ctx.socket(zmq.DEALER)
        dealer_socket.setsockopt_string(zmq.IDENTITY, '300')
        dealer_socket.connect(server_address)
        dealer_controller = zmq.Poller()
        dealer_controller.register(dealer_socket, zmq.POLLIN)

        neighbors = [ path.split(elm)[-1] for elm in glob(path.join(source, '*.jpg'))[:size] ]

        keep_interface = True 
        while keep_interface:
            incoming_events = dict(dealer_controller.poll(100))
            if dealer_socket in incoming_events:
                if incoming_events[dealer_socket] == zmq.POLLIN:
                    _, contents = dealer_socket.recv_multipart()
                    decoded_contents = json.loads(contents.decode())
                    if decoded_contents['global_status'] == 1:
                        duplicated = decoded_contents['response']['duplicated']
                        print(duplicated)
                    else:
                        print('The server was not able to handle this request')
                        print(decoded_contents['error_message'])
                        
            is_ok = input('do you wanna send new request ?')
            if is_ok == 'yes':
                image_name = input('choose an image name : ')
                request2send = json.dumps({
                    'candidate': image_name,
                    'neighbors': neighbors
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