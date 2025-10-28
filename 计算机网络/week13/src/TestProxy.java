interface IProxy {
    void submit();
}
class PersonA implements IProxy {
    @Override
    public void submit() {
        System.out.println("PersonA提交了⼀份报告");
    }
}
// 编写类PersonB
class PersonB implements IProxy {
    //被代理者的引⽤
    private IProxy m_Person;

    PersonB(IProxy person) {
        m_Person = person;
    }

    @Override
    public void submit() {
        before();
        m_Person.submit();
    }

    public void before() {
        System.out.println("PersonB加上抬头");
    }
}
public class TestProxy {
    public static void main(String[] args) {
// 构造⼀个PersonA对象
        PersonA personA = new PersonA();
// 构造⼀个代理，将personA作为参数传递进去
        IProxy proxy = new PersonB(personA);
// 由代理者来提交
        proxy.submit();
    }
}