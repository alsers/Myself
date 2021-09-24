
public class Named_Hello {

    public static void main (String[] args) {
        System.out.println("Arthur_Zhu is saying hello to you, java.");
    }
}


/*
introspection:
String[]大写； println中的ln； 每句后的分号；

And：
1. 编译后，每一个类都会有一个 .class （公共类只能有一个，但是其他类可以有无数个）
     比如：
     class dog{}
     class tiger{}
   运行后就会有dog.class, tiger.class
2. 文件名必须和公共类名字相同
 */


// 3. 非公共类也可以将main方法写入其中，并运行非public类
class dog {
    public static void main (String[] args){
        System.out.println("你好，狗子！");
    }

}

class tiger{
    public static void main (String[] args){
        System.out.println("你好，大猫!");
    }
}