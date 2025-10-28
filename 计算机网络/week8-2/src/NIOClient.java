import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.SelectionKey;
import java.nio.channels.Selector;
import java.nio.channels.SocketChannel;
import java.util.Iterator;
import java.util.Scanner;
import java.util.Set;

public class NIOClient {
    private static final int BYTE_LENGTH = 64;
    ByteBuffer writeBuffer = ByteBuffer.allocate(BYTE_LENGTH);
    ByteBuffer readBuffer = ByteBuffer.allocate(BYTE_LENGTH);

    public void start() throws IOException {
        int port = 9091;
        int connected=0;
        // 打开socket通道  
        SocketChannel sc = SocketChannel.open();
        // 设置为非阻塞
        sc.configureBlocking(false);
        // 连接服务器地址和端口
        sc.connect(new InetSocketAddress("127.0.0.1", port));
        // 创建选择器
        Selector selector = Selector.open();
        // 注册连接服务器socket的CONNECT事件
        sc.register(selector, SelectionKey.OP_CONNECT);

        Scanner scanner = new Scanner(System.in);
        // 保持连接，持续处理事件
        while (true) {
            // 该调用会阻塞，直至至少有一个事件发生
            // 比如上面 connect() 连接服务端成功就会触发 CONNECT 事件
            selector.select();
            // 获取发生事件的SelectionKey
            Set<SelectionKey> keys = selector.selectedKeys();

            Iterator<SelectionKey> keyIterator = keys.iterator();
            while (keyIterator.hasNext()) {

                SelectionKey key = keyIterator.next();
                // 连接事件
                if (key.isConnectable()&&connected==0) {
//                    System.out.println("OP_CONNECT");
                    sc.finishConnect();

//                    System.out.println("Server connected...");
//                    if (connected==0){
                    sc.write(ByteBuffer.wrap("Hello Server".getBytes()));
                    connected=1;
//                    }
                    // 注册WRITE事件，准备读取用户输入
                    sc.register(selector, SelectionKey.OP_READ);

                    // 写事件
                } else if (key.isWritable()) {
                    System.out.print("请输入消息:");
                    String message = scanner.next();
                    writeBuffer.clear();
                    writeBuffer.put(message.getBytes());
                    writeBuffer.flip();
                    sc.write(writeBuffer);
                    // 注册读操作，下一次读取
                    sc.register(selector, SelectionKey.OP_READ);

                    // 读事件
                } else if (key.isReadable()){
                    System.out.print("从服务端收到消息: ");
                    readBuffer.clear();
                    int numRead = sc.read(this.readBuffer);
                    System.out.println(new String(readBuffer.array(),0, numRead));
                    sc.register(selector, SelectionKey.OP_WRITE);
                }
                keyIterator.remove();
            }
        }
    }

    public static void main(String[] args) throws IOException {
        new NIOClient().start();
    }

}
