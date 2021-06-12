using System;

namespace Lab1Mo
{ 
    class Table
    {
        private Tuple<string,int>[] mas;
        private int n = 0;
        public void Add(Tuple<string, int> ma)
        {
            bool fl = true;
            n++;
            Array.Resize(ref mas, n);
            for (int i = 0; i < n - 1; i++)
            {
                if (ma.Item2 > mas[i].Item2)
                {
                    for (int j = 0; j < n - i - 1; j++)
                    {
                        mas[n - j - 1] = mas[n - j - 2];
                    }
                    mas[i] = ma;                    
                    fl = false;
                    break;
                }
            }
            if (fl)
            {
                mas[n-1] = ma;
            }
        }
        public Tuple<string, int> BinSearch(int ma)
        {
            int z = (n - 1) / 2, a = 0, b = n-1;
            if (mas[a].Item2 >= ma && mas[b].Item2 <= ma)
            {
                while ((b - a) != 0) 
                {                
                    if (ma > mas[z].Item2)
                    {
                        b = z;
                    }
                    else if (ma < mas[z].Item2)
                    {
                        a = z;
                        if (mas[b].Item2 == ma)
                        {
                            return mas[b];
                        }
                    }
                    else
                    {
                        return mas[z];
                    }
                    z = a + (b - a) / 2;
                }
                if (ma == mas[a].Item2) 
                {
                    return mas[a];
                }
                else
                {
                    Tuple<string, int> buf = Tuple.Create("0", 0);
                    return buf;
                }
            }
            else
            {
                Tuple<string, int> buf = Tuple.Create("0", 0);
                return buf;
            }
        }
        public void Delete(string ma, ref bool fl)
        {
            for (int i = 0; i < n; i++)
            {
                if (ma == mas[i].Item1)
                {
                    for (int j = i; j < n - 1; j++)
                    {
                        mas[j] = mas[j + 1];
                    }
                    fl = true;
                    break;
                }
            }
            if (fl)
            {
                Array.Resize(ref mas, n - 1);
            }
            n--;
        }
        public void Print()
        {
            string d;
            for (int i = 0; i < n; i++)
            {
                d = mas[i].Item1;
                d += " ";
                d += mas[i].Item2;
                Console.WriteLine(d, "/n");
            }
        }
    }
    class Program
    {
       
        static void Main(string[] args)
        {
            Table T = new Table();
            Console.WriteLine("Enter the size of new table:");
            int k = Convert.ToInt32(Console.ReadLine());
            for (int i = 0; i < k; i++)
            {
                Console.WriteLine("Enter name and value:");
                string s = Console.ReadLine();
                int x = Convert.ToInt32(Console.ReadLine());
                Tuple<string, int> z = Tuple.Create(s, x);
                T.Add(z);
            }
            Console.WriteLine("Your table:");
            T.Print();
            bool flag = true;
            while (flag)
            {
                Console.WriteLine("1. Append " + "2. Search " + "3. Delete " +
                "4. Finish\n");
                int p = Convert.ToInt32(Console.ReadLine());
                switch (p)
                {
                    case 1:
                        {
                            Console.WriteLine("Enter name and value:");
                            string s = Console.ReadLine();
                            int x = Convert.ToInt32(Console.ReadLine());
                            var z = Tuple.Create(s, x);
                            T.Add(z);
                            Console.WriteLine("Your table:");
                            T.Print();
                        }
                        break;
                    case 2:
                        {
                            Console.WriteLine("Enter name for search: ");
                            int s = Convert.ToInt32(Console.ReadLine());
                            Tuple<string, int> buf = T.BinSearch(s);
                            if (buf.Item1 == "0")
                            {
                                Console.WriteLine("There is no such element");
                            }
                            else
                            {
                                Console.WriteLine("Found element: ");
                                string d = buf.Item1;
                                d += " ";
                                d += buf.Item2;
                                Console.WriteLine(d, "/n");
                            }
                        }
                        break;
                    case 3:
                        {
                            Console.WriteLine("Enter name for delete: ");
                            string s = Console.ReadLine();
                            bool fl = false;
                            T.Delete(s,ref fl);
                            if (fl)
                            {
                                Console.WriteLine("Your table after deleting:");
                                T.Print();
                            }
                            else
                            {
                                Console.WriteLine("There is no such element");
                            }
                        }
                        break;
                    case 4:
                        {
                            flag = false;
                            Console.WriteLine("The end");
                        }
                        break;
                }
            }
            Console.ReadKey();
        }
    }
}
    

