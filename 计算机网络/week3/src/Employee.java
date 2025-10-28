import java.util.Calendar;

public abstract class Employee {
    protected String name;
    protected int birthday;
    abstract public int getSalary(int month);
}
class SalesEmployee extends Employee{
    protected int monthSales;
    protected float ticheng;
    SalesEmployee(String name,int monthSales,float ticheng,int birthday){
        this.monthSales=monthSales;
        this.ticheng=ticheng;
        this.name=name;
        this.birthday=birthday;
    }
    @Override
    public int getSalary(int month) {
        return 3000+ (int)(this.monthSales*this.ticheng);
    }
}
class SalariedEmployee extends Employee{
    SalariedEmployee(String name,int birthday){
        this.name=name;
        this.birthday=birthday;
    }
    @Override
    public int getSalary(int month) {
        if(month == this.birthday){
            return 3000+100;
        }
        return 3000;
    }
}
class HourlyEmployee extends Employee{
    protected int hours;
    protected int salaryPerHour=24;
    HourlyEmployee(String name,int hours,int salaryPerHour,int birthday){
        this.hours=hours;
        this.salaryPerHour=salaryPerHour;
        this.name=name;
        this.birthday=birthday;
    }
    @Override
    public int getSalary(int month) {
        return this.hours*this.salaryPerHour;
    }
}
