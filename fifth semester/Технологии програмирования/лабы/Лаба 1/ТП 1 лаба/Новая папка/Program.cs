using System;

namespace Lab1
{
    class FAL
    {
        public FAL(int[] mb)
        {
            ma = mb;
        }
        
        public static int[] Kon(int[] mb, int[] mc)
        {
            int[] mr = new int[mb.Length];
            for(int i = 0; i < mb.Length; i++)
            {
                if (mb[i] == 1 && mc[i] == 1)
                {
                    mr[i] = 1;
                }
                else
                {
                    mr[i] = 0;
                }
            }
            return mr;
        }
        public static int[] Diz(int[] mb, int[] mc)
        {
            int[] mr = new int[mb.Length];
            for (int i = 0; i < mb.Length; i++)
            {
                if (mb[i] == 1 || mc[i] == 1)
                {
                    mr[i] = 1;
                }
                else
                {
                    mr[i] = 0;
                }
            }
            return mr;
        }
        public static int[] Ne(int[] mb)
        {
            int[] mr = new int[mb.Length];
            for (int i = 0; i < mb.Length; i++)
            {
                if (mb[i] == 1)
                {
                    mr[i] = 0;
                }
                else
                {
                    mr[i] = 1;
                }
            }
            return mr;
        }
        public static int[] mod2(int[] mb, int[] mc)
        {
            int[] mr = new int[mb.Length];
            for(int i = 0; i < mb.Length; i++)
            {
                if (mb[i] == mc[i])
                {
                    mr[i] = 0;
                }
                else
                {
                    mr[i] = 1;
                }
            }
            return mr;
        }
        private int[] ma;

    }
    class Program
    {
        static void Main(string[] args)
        {
            int n;
            string nstr;
            Console.WriteLine("Write number of elements");
            nstr = Console.ReadLine();
            n = Convert.ToInt32(nstr);
            int[] ma = new int[n];
            int[] mb = new int[n];
            int[] mr = new int[n];
            Console.WriteLine("Write all elements 1st mas betwen 1 and 0");
            for(int i = 0; i < n; i++)
            {
                nstr = Console.ReadLine();
                ma[i] = Convert.ToInt32(nstr);
            }
            Console.WriteLine("Write all elements 2nd mas betwen 1 and 0");
            for (int i = 0; i < n; i++)
            {
                nstr = Console.ReadLine();
                mb[i] = Convert.ToInt32(nstr);
            }
            mr = FAL.Kon(ma, mb);
            Console.WriteLine("Conjunction: ");
            for(int i = 0; i < n; i++)
            {
                Console.WriteLine(mr[i]);
            }
            mr = FAL.Diz(ma, mb);
            Console.WriteLine("Disjunction: ");
            for (int i = 0; i < n; i++)
            {
                Console.WriteLine(mr[i]);
            }           
            mr = FAL.mod2(ma, mb);
            Console.WriteLine("Strict disjunction: ");
            for (int i = 0; i < n; i++)
            {
                Console.WriteLine(mr[i]);
            }
            mr = FAL.Ne(ma);
            Console.WriteLine("Inversion of 1st mas: ");
            for (int i = 0; i < n; i++)
            {
                Console.WriteLine(mr[i]);
            }
            mr = FAL.Ne(mb);
            Console.WriteLine("Inversion of 2nd mas: ");
            for (int i = 0; i < n; i++)
            {
                Console.WriteLine(mr[i]);
            }

        }
    }
}
