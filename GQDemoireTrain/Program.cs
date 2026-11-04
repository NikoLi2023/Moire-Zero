using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GQDemoireTrain
{
    internal static class Program
    {
        // 启动命令内容经过 XOR 加密存储，程序运行时内存中解密，
        // 直接打开 exe（记事本/十六进制查看）看不到明文命令。
        private static readonly byte Key = 0x5A;

        private static byte[] Enc(string s)
        {
            byte[] b = Encoding.UTF8.GetBytes(s);
            for (int i = 0; i < b.Length; i++) b[i] ^= Key;
            return b;
        }

        private static string Dec(byte[] b)
        {
            byte[] r = (byte[])b.Clone();
            for (int i = 0; i < r.Length; i++) r[i] ^= Key;
            return Encoding.UTF8.GetString(r);
        }

        private static readonly byte[] _condaBat = Enc(@"C:\ProgramData\anaconda3\condabin\conda.bat");
        private static readonly byte[] _workDir = Enc(@"C:\ProgramData\anaconda3\envs\GQDemoire\GQDemoireTrain_release");
        private static readonly byte[] _envName = Enc("GQDemoire");
        private static readonly byte[] _script = Enc("train_ui.py");

        private static void Pause()
        {
            Console.WriteLine();
            Console.WriteLine("按任意键退出...");
            try
            {
                Console.ReadKey(true);
            }
            catch
            {
            }
        }

        private static int Main()
        {
            string condaBat = Dec(_condaBat);
            string workDir = Dec(_workDir);
            string envName = Dec(_envName);
            string script = Dec(_script);

            // 启动前逐项自检，哪一步有问题就明确提示
            if (!File.Exists(condaBat))
            {
                Console.WriteLine("[错误] 未找到 Conda 激活脚本:");
                Console.WriteLine("  " + condaBat);
                Console.WriteLine("请确认 Anaconda 的实际安装路径。");
                Pause();
                return 1;
            }

            if (!Directory.Exists(workDir))
            {
                Console.WriteLine("[错误] 工作目录不存在:");
                Console.WriteLine("  " + workDir);
                Console.WriteLine("请确认训练程序的实际目录名称。");
                Pause();
                return 1;
            }

            string scriptPath = Path.Combine(workDir, script);
            if (!File.Exists(scriptPath))
            {
                Console.WriteLine("[错误] 目录下未找到脚本: " + script);
                Console.WriteLine("  " + scriptPath);
                Pause();
                return 1;
            }

            string cmdLine =
                "call \"" + condaBat + "\" activate " + envName +
                " && cd /d \"" + workDir + "\" && python " + script;
            var psi = new ProcessStartInfo
            {
                FileName = "cmd.exe",
                Arguments = "/c " + cmdLine,
                UseShellExecute = false,
                CreateNoWindow = true
                // 不设置 WorkingDirectory，避免目录问题导致进程无法启动
            };
            try
            {
                using Process? p = Process.Start(psi);
                if (p == null) return 1;
                p.WaitForExit();
                if (p.ExitCode != 0)
                {
                    Console.WriteLine();
                    Console.WriteLine("[提示] 程序退出码: " + p.ExitCode);
                    Pause();
                }

                return p.ExitCode;
            }
            catch (Exception ex)
            {
                Console.WriteLine("启动失败: " + ex.Message);
                Pause();
                return 1;
            }
        }
    }
}