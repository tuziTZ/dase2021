import java.io.*;
import java.net.*;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.SocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.*;
import java.util.Arrays;
import java.util.Iterator;

public class NIOWebServer {
    private static final int PORT = 8083;
    private ServerSocketChannel serverSocketChannel;
    private Selector selector;

    public static void main(String[] args) {
        NIOWebServer server = new NIOWebServer();
        server.start();
    }

    public void start() {
        try {
            selector = Selector.open();
            serverSocketChannel = ServerSocketChannel.open();
            serverSocketChannel.bind(new InetSocketAddress(PORT));
            serverSocketChannel.configureBlocking(false);
            serverSocketChannel.register(selector, SelectionKey.OP_ACCEPT);
            System.out.println("Server started. Listening at port " + PORT);

            while (true) {
                selector.select();

                Iterator<SelectionKey> iterator = selector.selectedKeys().iterator();
                while (iterator.hasNext()) {
                    SelectionKey key = iterator.next();
                    if (key.isAcceptable()) {
                        SocketChannel clientChannel = serverSocketChannel.accept();
                        clientChannel.configureBlocking(false);
                        clientChannel.register(selector, SelectionKey.OP_READ);
                        System.out.println("Accepted connection from " + clientChannel.getRemoteAddress());
                    } else if (key.isReadable()) {
                        SocketChannel clientChannel = (SocketChannel) key.channel();
                        ByteBuffer readBuffer = ByteBuffer.allocate(1024);
                        int bytesRead = clientChannel.read(readBuffer);
                        if (bytesRead == -1) {
                            System.out.println("Connection closed by client.");
                            clientChannel.close();
                            continue;
                        }

                        readBuffer.flip();
                        String request = new String(readBuffer.array(), 0, bytesRead);
//                        System.out.println("Received request: " + request);
                        new Thread(new RequestHandler(request,clientChannel)).start();

                    }

                    iterator.remove();
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
class RequestHandler implements Runnable {
    private String request;
    private SocketChannel clientChannel;

    public RequestHandler(String request,SocketChannel clientChannel) {
        this.request = request;
        this.clientChannel=clientChannel;
    }

    @Override
    public void run() {
        try {
            // 处理请求内容
            String[] tokens = request.split(" ");
            String method = tokens[0].toUpperCase();
            String resource = tokens[1];
            if (method.equals("GET")||method.equals("POST")) {
                if (resource.contains("/index.html")) {
                    File file = new File("D:/Javaprojects/networkFinal/src/index.html");
                    if (file.exists()) {
                        clientChannel.write(ByteBuffer.wrap("HTTP/1.1 200 OK\r\n".getBytes()));
                        clientChannel.write(ByteBuffer.wrap("Content-Type: text/html; charset=UTF-8\r\n".getBytes()));
                        clientChannel.write(ByteBuffer.wrap("\r\n".getBytes()));

                        //读文件
                        FileInputStream fileInputStream = null;
                        try {
                            fileInputStream = new FileInputStream(file);
                        } catch (FileNotFoundException e) {
                            throw new RuntimeException(e);
                        }
                        FileChannel channel=fileInputStream.getChannel();
    //                                    long size=channel.size();
                        byte[] fileBuffer = new byte[1024];
                        ByteBuffer fBuffer=ByteBuffer.allocate(1024);
                        int len;

                        while ((len=channel.read(fBuffer)) != -1) {
                            fBuffer.flip();
                            fBuffer.get(fileBuffer,0,len);
                            clientChannel.write(ByteBuffer.wrap(new String(fileBuffer,0,len).getBytes()));
    //                                        System.out.println(new String(fileBuffer,0,len));
                            fBuffer.clear();
                        }
                        channel.close();
                        fileInputStream.close();
                        clientChannel.close();
                    } else {
    //                        System.out.println("file not exists");
                        clientChannel.write(ByteBuffer.wrap("HTTP/1.1 404 Not Found\r\n".getBytes()));
                        clientChannel.write(ByteBuffer.wrap("Content-Type: text/plain; charset=UTF-8\r\n".getBytes()));
                        clientChannel.write(ByteBuffer.wrap("\r\n".getBytes()));
                        clientChannel.write(ByteBuffer.wrap("File not found.".getBytes()));
                        clientChannel.close();
                    }
                } else if (resource.equals("/shutdown")) {
                    clientChannel.write(ByteBuffer.wrap("HTTP/1.1 200 OK\r\n".getBytes()));
                    clientChannel.write(ByteBuffer.wrap("Content-Type: text/plain; charset=UTF-8\r\n".getBytes()));
                    clientChannel.write(ByteBuffer.wrap("\r\n".getBytes()));
                    clientChannel.write(ByteBuffer.wrap("Server is shutting down...".getBytes()));
                    clientChannel.close();
                    System.exit(0);
                } else {
                    clientChannel.write(ByteBuffer.wrap("HTTP/1.1 404 Not Found\r\n".getBytes()));
                    clientChannel.write(ByteBuffer.wrap("Content-Type: text/plain; charset=UTF-8\r\n".getBytes()));
                    clientChannel.write(ByteBuffer.wrap("\r\n".getBytes()));
                    clientChannel.write(ByteBuffer.wrap("404 not found.".getBytes()));
                    clientChannel.close();
                }
            }
            else {
                clientChannel.write(ByteBuffer.wrap("HTTP/1.1 405 Method Not Allowed\r\n".getBytes()));
                clientChannel.write(ByteBuffer.wrap("Content-Type: text/plain; charset=UTF-8\r\n".getBytes()));
                clientChannel.write(ByteBuffer.wrap("\r\n".getBytes()));
                clientChannel.write(ByteBuffer.wrap("Method not allowed.".getBytes()));
                clientChannel.close();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}