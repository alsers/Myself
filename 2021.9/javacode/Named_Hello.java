
public class Named_Hello {

    public static void main (String[] args) {
        System.out.println("Arthur_Zhu is saying hello to you, java.");
    }
}


/*
introspection:
String[]��д�� println�е�ln�� ÿ���ķֺţ�

And��
1. �����ÿһ���඼����һ�� .class ��������ֻ����һ���������������������������
     ���磺
     class dog{}
     class tiger{}
   ���к�ͻ���dog.class, tiger.class
2. �ļ�������͹�����������ͬ
 */


// 3. �ǹ�����Ҳ���Խ�main����д�����У������з�public��
class dog {
    public static void main (String[] args){
        System.out.println("��ã����ӣ�");
    }

}

class tiger{
    public static void main (String[] args){
        System.out.println("��ã���è!");
    }
}