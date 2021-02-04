from socket import *

serversocket = socket(AF_INET, SOCK_STREAM)
serversocket.bind(('localhost', 9000))
serversocket.listen(5)

while(1):
    (clientsocket, address) = serversocket.accept()
    output = """HTTP/1.0 200 OK\n
        Content-Type: text/html\n
        \n
        <html><body>Hello</body></html>\n"""

    clientsocket.sendall(output.encode("utf-8"))

    clientsocket.shutdown(SHUT_WR)
    clientsocket.close()
serversocket.close()
