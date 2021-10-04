import zmq 
import click 
import numpy as np
import json 

import time 

from os import path 
from loguru import logger 
from glob import glob 

@click.command()
@click.option('--source', help='path 2 dataset')
@click.option('--server_address', help='address of the remote app', required=True)
def interface(source, server_address):
    try:
        ctx = zmq.Context()
        
        dealer_socket = ctx.socket(zmq.DEALER)
        dealer_socket.setsockopt_string(zmq.IDENTITY, '300')
        dealer_socket.connect(server_address)
        dealer_controller = zmq.Poller()
        dealer_controller.register(dealer_socket, zmq.POLLIN)

        file_paths = sorted(glob(path.join(source, '*.jpg')))
        print('nb images :', len(file_paths))

        cursor = 0 
        keep_interface = True 
        while keep_interface:
            incoming_events = dict(dealer_controller.poll(100))
            if dealer_socket in incoming_events:
                if incoming_events[dealer_socket] == zmq.POLLIN:
                    _, contents = dealer_socket.recv_multipart()
                    decoded_contents = json.loads(contents.decode())
                    if decoded_contents['global_status'] == 1:
                        if decoded_contents['response']['local_status'] == 1:
                            fingerprint = np.asarray(decoded_contents['response']['fingerprint'])
                            print(fingerprint[:30], fingerprint.shape)
                    else:
                        print('The server was not able to handle this request')
                        print(decoded_contents['error_message'])

            current_path = file_paths[cursor]
            _, image_name = path.split(current_path)
            request2send = json.dumps({
                'request_id': cursor,
                'image_name': image_name
            }).encode()

            dealer_socket.send_multipart([b'', request2send])
            cursor = cursor + 1 
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