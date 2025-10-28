import java.util.Date;


public class Watch {
    public Date startTime;
    public Date endTime;
    public void start(){
        this.startTime= new Date();
    }
    public void stop(){
        this.endTime= new Date();
    }
}
class StopWatch extends Watch{
    public long getElapsedTime(){
        return super.endTime.getTime()-super.startTime.getTime();
    }

}
