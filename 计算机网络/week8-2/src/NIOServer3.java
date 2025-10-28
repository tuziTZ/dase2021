import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.*;
import java.util.Iterator;
import java.util.Set;

public class NIOServer3 {

    // Server全局共用一个选择器
    private Selector selector;

    // 所有连接共用一个读、写缓冲区
    // PS：所以每次读写前都要先清空缓冲区（clear）
    private ByteBuffer readBuffer = ByteBuffer.allocate(1024);
    private ByteBuffer sendBuffer = ByteBuffer.allocate(1024);

    public void start() throws IOException {
        // 打开服务器套接字通道
        ServerSocketChannel ssc = ServerSocketChannel.open();
        // 必须配置为非阻塞，该Channel才能往selector上注册，否则会报错，selector模式本身就是非阻塞模式
        ssc.configureBlocking(false);
        // 进行服务的绑定
        ssc.bind(new InetSocketAddress( 9091));

        // 创建Selector对象
        selector = Selector.open();
        // 向Selector中注册感兴趣的事件（这里的ACCEPT就是新连接发生时所产生的事件）
        // 注：对于ServerSocketChannel 通道来说，我们唯一可以指定的参数就是OP_ACCEPT
        ssc.register(selector, SelectionKey.OP_ACCEPT);

        // 当前只有 server 这一个线程，所以只要该线程不中断就能一直提供服务
        while (!Thread.currentThread().isInterrupted()) {
            // 该调用会阻塞，直至至少有一个事件发生
            // 1.当有客户端来连接时，就会触发 ServerSocketChannel 的 ACCEPT 事件
            // 2.当客户端发送来消息时，就会触发 ServerSocketChannel 的 READ 事件
            // 3.当客户端读取了发送的消息时，就会触发 ServerSocketChannel 的 WRITE 事件
            selector.select();
            // 获取发生事件的 SelectionKey
            Set<SelectionKey> keys = selector.selectedKeys();
            // 迭代所有事件
            Iterator<SelectionKey> keyIterator = keys.iterator();
            while (keyIterator.hasNext()) {
                // 拿到当前事件，根据事件类型进行相应处理
                SelectionKey key = keyIterator.next();
                if (!key.isValid()) {
                    continue;
                }
                if (key.isAcceptable()) {
                    accept(key);
                } else if (key.isReadable()) {
                    read(key);
                } else if (key.isWritable()) {
                    write(key);
                }
                // 丢弃已经处理过的事件
                keyIterator.remove();
            }
        }
    }

    // 接收请求
    private void accept(SelectionKey key) throws IOException {
        // 通过key拿到注册当前事件的Channel（因为是ACCEPT，所以只能是ServerSocketChannel ）
        ServerSocketChannel ssc = (ServerSocketChannel) key.channel();
        // accept返回一个包含新连接的SocketChannel，将会为当前client提供服务
        // NIO非阻塞体现：此处accept方法是阻塞的，但是这里因为是发生了连接事件，所以这个方法会马上执行完，不会阻塞
        SocketChannel clientChannel = ssc.accept();
        clientChannel.configureBlocking(false);

        // 将该socketChannel注册到Selector，绑定上READ事件
        clientChannel.register(selector, SelectionKey.OP_READ);
        System.out.println("a new client connected "+ clientChannel.getRemoteAddress());
    }

    // 读事件
    private void read(SelectionKey key) throws IOException {
        // 通过key拿到当前连接的 SocketChannel
        SocketChannel socketChannel = (SocketChannel) key.channel();

        // 先清除缓冲区的数据，为本次要读的数据做准备
        this.readBuffer.clear();

        // 将通道中的数据读到缓冲区 readBuffer
        // numRead 记录读了多少字符
        int numRead = socketChannel.read(this.readBuffer);
        // 通过 buffer.array() 拿到buffer中的数组（没必要buffer.filp）
        String str = new String(readBuffer.array(), 0, numRead);
        System.out.println(socketChannel.getRemoteAddress() + " send message: " + str);

        // 将当前SocketChannel再注册上WRITE事件
        socketChannel.register(selector, SelectionKey.OP_WRITE);
    }

    // 写事件
    private void write(SelectionKey key) throws IOException {
        SocketChannel channel = (SocketChannel) key.channel();

        // 先清除缓冲区的数据，为本次写数据做准备
        sendBuffer.clear();
        sendBuffer.put("Server ACK".getBytes());
        // 这里必须 filp()，因为 put() 后 position 等于消息的length，所以client拿到后无法再从这个缓冲区读出数据
        sendBuffer.flip();

        // 将缓冲区的数据写到SocketChannel
        channel.write(sendBuffer);

        // 给当前channel再注册上READ事件，为下次读它发送的内容做准备
        channel.register(selector, SelectionKey.OP_READ);
    }

    public static void main(String[] args) throws IOException {
        System.out.println("Server started...");
        new NIOServer3().start();
    }
}
