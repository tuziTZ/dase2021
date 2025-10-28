import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.net.SocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.SelectionKey;
import java.nio.channels.Selector;
import java.nio.channels.ServerSocketChannel;
import java.nio.channels.SocketChannel;
import java.nio.charset.StandardCharsets;
import java.util.Iterator;
import java.util.Scanner;
import java.util.Set;
public class NIOServer2 {
    private static final int BYTE_LENGTH = 64;
    private Selector selector;
    public static void main(String[] args) throws IOException {
        try {
            new NIOServer2().startServer();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    private void startServer() throws IOException {
        this.selector = Selector.open();
// ServerSocketChannel与serverSocket类似
        ServerSocketChannel serverSocket = ServerSocketChannel.open();
        serverSocket.socket().bind(new InetSocketAddress(9091));
// 设置⽆阻塞
        serverSocket.configureBlocking(false);
// 将channel注册到selector
        serverSocket.register(this.selector, SelectionKey.OP_ACCEPT);

        System.out.println("服务端已启动");
        for (;;) {
// 操作系统提供的⾮阻塞I/O
            int readyCount = selector.select();
            if (readyCount == 0) {
                continue;
            }
// 处理准备完成的fd
            Set<SelectionKey> readyKeys = selector.selectedKeys();
            Iterator<SelectionKey> iterator = readyKeys.iterator();
            while (iterator.hasNext()) {
                SelectionKey key = iterator.next();
                iterator.remove();
                if (!key.isValid()) {
                    continue;
                }
                if (key.isAcceptable()) {
                    this.accept(key);
                } else if (key.isReadable()) {
                    this.read(key);
                } else if (key.isWritable()) {
                    this.write(key);
                }
            }
        }
    }
    private void accept(SelectionKey key) throws IOException {
        ServerSocketChannel serverChannel = (ServerSocketChannel) key.channel();
        SocketChannel channel = serverChannel.accept();
        channel.configureBlocking(false);
        Socket socket = channel.socket();
        SocketAddress remoteAddr = socket.getRemoteSocketAddress();
        System.out.println("已连接: " + remoteAddr);
// 监听读事件
        channel.register(this.selector, SelectionKey.OP_READ);

    }
    private void read(SelectionKey key) throws IOException {
        SocketChannel channel = (SocketChannel) key.channel();
        ByteBuffer buffer = ByteBuffer.allocate(BYTE_LENGTH);
        int numRead = -1;
        numRead = channel.read(buffer);
        if (numRead == -1) {
            Socket socket = channel.socket();
            SocketAddress remoteAddr = socket.getRemoteSocketAddress();
            System.out.println("连接关闭: " + remoteAddr);
            channel.close();
            key.cancel();
            return;
        }
        byte[] data = new byte[numRead];
        System.arraycopy(buffer.array(), 0, data, 0, numRead);
        System.out.println("从客户端收到消息: " + new String(data));
        channel.register(this.selector, SelectionKey.OP_WRITE);
    }
    private void write(SelectionKey key) throws IOException {
//        SocketChannel channel = (SocketChannel) key.channel();
//        ByteBuffer sendBuffer = ByteBuffer.allocate(BYTE_LENGTH);
//        sendBuffer.clear();
//        Scanner sc=new Scanner(System.in);
//        String serverWord=sc.next();
//        sendBuffer.put(serverWord.getBytes());
//        sendBuffer.flip();
//        channel.write(sendBuffer);
//        channel.register(this.selector, SelectionKey.OP_READ);
        SocketChannel channel = (SocketChannel) key.channel();
        ByteBuffer sendBuffer = ByteBuffer.allocate(BYTE_LENGTH);
        sendBuffer.clear();
        sendBuffer.put("收到！".getBytes());
        sendBuffer.flip();
        channel.write(sendBuffer);
        channel.register(selector, SelectionKey.OP_READ);
    }
}