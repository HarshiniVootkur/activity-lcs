#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <iostream>
#include <unistd.h>

#include <vector>
#include <chrono>
#include <omp.h>

#ifdef __cplusplus
extern "C" {
#endif

  void generateLCS(char* X, int m, char* Y, int n);
  void checkLCS(char* X, int m, char* Y, int n, int result);

#ifdef __cplusplus
}
#endif

int lcs(char *X, int m, char *Y, int n, int nbthreads)
{
  //omp_set_num_threads(nbthreads);
  int large = std::max(n,m);
  int arr[large+1][large+1];

  #pragma omp parallel
  {
    #pragma omp for
    {
      for(int i=0;i<=large;i++)
	{
	  arr[i][0]=0;
	}
      #pragma omp for
      {
	for(int j=0;j<=large;j++)
	  arr[0][j]=0;
      }

      #pragma omp parallel
      {
	#pragma omp single
	{
	  int c=n/nbthreads;
	  for(int k=1;k<=m;k++)
	    {
	      if(X[k-1] == Y[k-1])
		  arr[k][k]=arr[k-1][k-1]+1;
	      else
		arr[k][k]=std::max(arr[k][k-1],arr[k-1][k]);

#pragma omp task shared(X,Y,arr,k,large)
	      {
#pragma omp parallel for schedule(guided,c)
		for(int j=k;j<=large;j++)
		  {
		    if(X[k-1] == Y[j])
		      arr[k][j]=arr[k-1][j-1]+1;
		    else
		      arr[k][j]=std::max(arr[k][j-1],arr[k-1][j]);
		  }
	      }

#pragma omp task shared(X,Y,arr,k,large)
	      {
#pragma omp parallel for schedule(guided,c)
		for(int i=k;i<=large;i++)
		  {
		    if(X[i]==Y[k-1])
		      arr[i][k]=arr[i-1][k-1]+1;
		    else
		      arr[i][k]=std::max(arr[i][k-1],arr[i-1][k]);
		  }
	      }
	      #pragma omp taskwait
	    }
	}
      }
    }
    return arr[m][n];
  }
}
				   
int main (int argc, char* argv[])
{

  if (argc < 4)
    {
      std::cerr<<"usage: "<<argv[0]<<" <m> <n> <nbthreads>"<<std::endl;
    return -1;
  }

  int m = atoi(argv[1]);
  int n = atoi(argv[2]);
  int nbthreads = atoi(argv[3]);

  // get string data 
  char *X = new char[m];
  char *Y = new char[n];
  generateLCS(X, m, Y, n);

  
  //insert LCS code here.
  int result = -1; // length of common subsequence

  std::chrono::time_point<std::chrono::system_clock>startTime=std::chrono::system_clock::now();

  int Lcs = lcs(X,m,Y,n,nbthreads);

   std::chrono::time_point<std::chrono::system_clock>endTime=std::chrono::system_clock::now();

  std::chrono::duration<double> totalTime = endTime-startTime;

  checkLCS(X, m, Y, n, result);
  std::cerr<<totalTime.count()<<std::endl;


  return 0;
}
