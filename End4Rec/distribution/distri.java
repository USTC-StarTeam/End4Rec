// mpirun -n 4 java -cp . distributed.launcher.DistributedLauncher 4 config.json eval_data.csv
package distributed.launcher;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.IOException;

public class DistributedLauncher {

    public static void main(String[] args) {
        if(args.length < 3) {
            System.out.println("Usage: java DistributedLauncher <device_num> <config_file> <eval_data_file>");
            System.exit(1);
        }
        int deviceNum = Integer.parseInt(args[0]);
        String configFile = args[1];
        String evalDataFile = args[2];
        DistributedLauncher launcher = new DistributedLauncher();
        try {
            launcher.launchCluster(deviceNum, configFile);
            launcher.startTraining(deviceNum, configFile);
            launcher.mergeCheckpoints(deviceNum, "./ckpt/", "./ckpt/merged_ckpt.ckpt");
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
    }

    public void launchCluster(int deviceNum, String configFile) throws IOException, InterruptedException {
        String launchCmd = String.format("mpirun -n %d python launch_cluster.py --device_target GPU --device_num %d --config_file %s", deviceNum, deviceNum, configFile);
        System.out.println("Launching cluster: " + launchCmd);
        int exitCode = runCommand(launchCmd);
        if(exitCode != 0) {
            System.out.println("Cluster launch failed with exit code " + exitCode);
            System.exit(exitCode);
        }
        int barrier = performBarrier();
        System.out.println("Barrier result: " + barrier);
    }

    public void startTraining(int deviceNum, String configFile) throws IOException, InterruptedException {
        String trainingCmd = String.format("mpirun -n %d python main.py %s", deviceNum, configFile);
        System.out.println("Starting training: " + trainingCmd);
        int exitCode = runCommand(trainingCmd);
        if(exitCode != 0) {
            System.out.println("Training failed with exit code " + exitCode);
            System.exit(exitCode);
        }
    }

    public void mergeCheckpoints(int deviceNum, String ckptDir, String outputCkpt) throws IOException, InterruptedException {
        String mergeCmd = String.format("python merge_checkpoints.py --ckpt_dir %s --output_ckpt %s", ckptDir, outputCkpt);
        System.out.println("Merging checkpoints: " + mergeCmd);
        int exitCode = runCommand(mergeCmd);
        if(exitCode != 0) {
            System.out.println("Checkpoint merge failed with exit code " + exitCode);
            System.exit(exitCode);
        }
    }

    public int runCommand(String command) throws IOException, InterruptedException {
        ProcessBuilder builder = new ProcessBuilder(command.split(" "));
        builder.redirectErrorStream(true);
        Process process = builder.start();
        BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
        String line;
        while((line = reader.readLine()) != null) {
            System.out.println(line);
        }
        int exitCode = process.waitFor();
        return exitCode;
    }

    public int performBarrier() throws IOException, InterruptedException {
        String barrierCmd = "python barrier_demo.py";
        System.out.println("Performing distributed barrier: " + barrierCmd);
        int exitCode = runCommand(barrierCmd);
        if(exitCode != 0) {
            System.out.println("Barrier command failed with exit code " + exitCode);
            System.exit(exitCode);
        }
        return 1;
    }
}
