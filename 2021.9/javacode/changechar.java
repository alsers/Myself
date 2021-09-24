// changechar
public class changechar {

    public static void main(String[] args){
        System.out.println("北京天津河北");
        // "\t"为制表符
        System.out.println("北京\t天津\t河北");
        // "\n"为换行符   
        System.out.println("北京\n天津\n河北"); 
        // "\\" 输出一个 "\"; 若要打出两个"\\"则需要4个"\"，因为一个转义符“\”后对应一个“\”
        System.out.println("北京\\天津\\河北"); 
        System.out.println("北京\\\\天津\\\\河北"); 
        // 输出 ""，跟python不一样，不能用单双引号区别，java只能识别双引号，所以需要转义"\"
        System.out.println("老韩说：\"要好好学习，才有饭吃\"");
        // "\r"：一个回车
        // 解读：
        // 1. 输出 北儿京儿的儿爷儿
        // 2. \r 表示回车，但是由于没有换行符，所以光标由最后一个字符后面 变为 输出行的首字符前 （即：“北”前面）
        // 3. \r 后的字符串将覆盖改行原先对应位置的字符，并进行最后输出
        System.out.println("北儿京儿的儿爷儿\r佩奇");

        // Practice
        System.out.println("书名\t作者\t价格\t销量\t\n\r三国\t罗贯中\t120\t1000");
    }
}