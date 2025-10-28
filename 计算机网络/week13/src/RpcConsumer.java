import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.util.Arrays;

public class RpcConsumer {
    public static void main(String[] args) {
        IProxy2 iProxy2 = (IProxy2) Proxy.newProxyInstance(
                IProxy2.class.getClassLoader(), new Class<?>[]{IProxy2.class}, new
                        iProxy2Handler()
        );
        System.out.println(iProxy2.sayHi("alice alice"));
    }
}
class iProxy2Handler implements InvocationHandler {
    @Override
    public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
        Socket socket = new Socket();
        socket.connect(new InetSocketAddress(9091));
        ObjectOutputStream os = new ObjectOutputStream(socket.getOutputStream());
// rpc提供⽅和调⽤⽅之间协商的报⽂格式和序列化规则
        os.writeUTF(method.getName());
        os.writeObject(method.getParameterTypes());
        os.writeInt(args.length);
        for (Object obj : args) {
            os.writeObject(obj);
        }
        return new ObjectInputStream(socket.getInputStream()).readObject();
    }
}