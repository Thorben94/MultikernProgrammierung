#include "image.h"
#include "immintrin.h"
#include <iostream>
#include <string>

__m256i vec_128 = _mm256_set1_epi8(static_cast<uint8_t>(128));

// vectoren dividieren mit tricks
__m256i _mm256_div2_epi8(__m256i &v) {
  __m256i srai1 = _mm256_srli_epi16(v, 1);
  srai1 = _mm256_andnot_si256(vec_128, srai1);    //vektoren sparen durch mehrfachdeutung
  return srai1;
}

// Sobel stufe 1 für einen Vektor aka 32 einträge
__m256i vectorSobel1(uint8_t *vecStart)  {
  // Load data l
  __m256i vecl = _mm256_loadu_si256(reinterpret_cast<__m256i *>(vecStart));
  // divide by two
  vecl = _mm256_div2_epi8(vecl);
  // Load data r
  __m256i vecr = _mm256_loadu_si256(reinterpret_cast<__m256i *>(vecStart + 2));
  // divide by two
  vecr = _mm256_div2_epi8(vecr);
  // add 128 base to vecl
  vecl = _mm256_add_epi8(vecl, vec_128);
  // subtract the values
  __m256i res = _mm256_sub_epi8(vecl, vecr);

  return res;
}

//Sobel stufe 1 für einen Punkt
uint8_t pointSobel1(Image &in, int i, int j) {
  // l = p(i - 1, j), r = p(i + 1, j)
  unsigned char l, r;

  // Default to zero if we are on the edge.
  if (i == 0)
    l = 0;
  else
    l = in.get(i - 1, j);

  // Default to zero if we are on the edge.
  if (i + 1 >= in.width())
    r = 0;
  else
    r = in.get(i + 1, j);

  return 128 + (l / 2) - (r / 2);
 
}

// Größe meines Filterkopfs
const int outStreams = 8;
__m256i sobel1results[outStreams + 2];
//__m256i sobel2results[outStreams];

void sobel2streamwriteback(Image &tmp, __m256i (&sobel1results)[outStreams + 2], int I, int j)  {
  for (int i = 0; i < outStreams + 2; i++)
  {
    sobel1results[i] = _mm256_div2_epi8(sobel1results[i]);
  }
  for (int i = 0; i < outStreams; i++)
  {
    __m256i res1 = _mm256_add_epi8(_mm256_div2_epi8(sobel1results[i]), _mm256_div2_epi8(sobel1results[i + 2]));
    // sobel2results[i] = _mm256_add_epi8(res1, sobel1results[i + 1]);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(&tmp.get(I, j + i)), _mm256_add_epi8(res1, sobel1results[i + 1]));
  }
}

void mergeSobel(Image &in, Image &tmp) {
  int I = 0;
  int J = 0;
  for (; J <= in.height() - outStreams; J += outStreams) {
    I = 0;
    for (; I + 32 + 2 <= in.width(); I += 32) {
      // top vector sobel stage 1
      if (J == 0)
      {
        sobel1results[0] = vec_128;
      }
      else
      {
        sobel1results[0] = vectorSobel1(&in.get(I, J - 1));
      }
      // mid section sobel stage 1
      for (int i = 1; i < outStreams + 2 - 1; i++)
      {
        sobel1results[i] = vectorSobel1(&in.get(I, J + i - 1));
      }
      // bot vector sobel stage 1
      if (J + outStreams >= in.height())
      {
        sobel1results[outStreams + 1] = vec_128;
      }
      else
      {
        sobel1results[outStreams + 1] = vectorSobel1(&in.get(I, J + outStreams));
      }
      // sobel stage 2
      sobel2streamwriteback(tmp, sobel1results, I + 1, J);
    }
  }
  // Jetzt kann unten noch was fehlen... -> hadle cutoff bottom
  if (J < in.height())
  {
    for (int j = J; j < in.height(); j++) {
      I = 0;
      for (; I + 32 + 2 <= in.width(); I += 32) {
        // top vector sobel stage 1
        __m256i top;
        if (j == 0)
        {
          top = vec_128;
        }
        else
        {
          top = vectorSobel1(&in.get(I, j - 1));
        }
        // mid vector sobel stage 1
        __m256i mid = vectorSobel1(&in.get(I, j));
        // bot vector sobel stage 1
        __m256i bot;
        if (j + 1 >= in.height())
        {
          bot = vec_128;
        }
        else
        {
          bot = vectorSobel1(&in.get(I, j + 1));
        }
        // sobel stage 2
        // divide down
        top = _mm256_div2_epi8(top);
        mid = _mm256_div2_epi8(mid);
        bot = _mm256_div2_epi8(bot);

        // add up
        __m256i res1 = _mm256_add_epi8(_mm256_div2_epi8(bot), mid);
        __m256i res = _mm256_add_epi8(_mm256_div2_epi8(top), res1);
        // store back into memory
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(&tmp.get(I + 1, j)), res);
      }
    }
  }
  
  //handle cutoff to the right
  for (int j = 0; j < in.height(); j++)
  {
    for (int i = I; i < in.width(); i++)
    {
      // l = p(i, j - 1), r = p(i, j + 1)
      unsigned char l, r;

      // Default to zero if we are on the edge.
      if (j == 0) {
        l = 0;
      }
      else
      {
        //l = in.get(i, j - 1);
        l = pointSobel1(in, i, j - 1);
      }
      // Default to zero if we are on the edge.
      if (j + 1 >= in.height())
      {
        r = 0;
      }
      else
      {
        r = pointSobel1(in, i, j + 1);
      }
      tmp.get(i, j) = (pointSobel1(in, i, j) / 2) + (l / 4) + (r / 4);
    }
  }
}

int main()
{ 
  // Read filenames.
  std::string fileIn, fileOut;
  std::cin >> fileIn >> fileOut;


  // Load image from disk.
  Image in(fileIn);
  // Temporary image for swapping.
  Image tmp(in.width(), in.height());
  
  mergeSobel(in, tmp);
  
  tmp.save(fileOut);

  return 0;
}
