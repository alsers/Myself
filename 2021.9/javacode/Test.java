// 老朱写的第一个Java程序
// win10 Java 中文编译乱码解决：编译时 javac -encoding utf-8 Test.java  （.java文件不能直接运行，需要javac先编译）
// 运行时：java Test    （此时不用打后缀）

public class Test {
    public static void main(String[] args) {
        int res = 1 + 1;
        //显示
        System.out.println("结果" + res);
    }
}