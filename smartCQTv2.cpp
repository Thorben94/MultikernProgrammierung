#include <iostream>
#include <vector>
#include <chrono>
#include <functional>

std::vector<double> sideLengthVector;

class Timer
{
public:
	std::chrono::time_point<std::chrono::steady_clock> tp;
	
	void start() {
		tp = std::chrono::steady_clock::now();
	}

	void stop() {
		std::chrono::milliseconds dur = std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::steady_clock::now() - tp);
		std::cout << "Time betweem start() and stop(): " << dur.count() << " ms\n";
	}
};

//source: Materials for UB2 
class Point {
public:
	double x;
	double y;

	Point()	{}

	Point(double xIn, double yIn)	{
		x = xIn;
		y = yIn;
	}

	~Point(){}
};

//source: Materials for UB2 
class Circle {
public:
	Point mid;
	double radius;
	
	bool contains(Point p) {
		if ((mid.x - p.x) * (mid.x - p.x) + (mid.y - p.y) * (mid.y - p.y) <=
			radius * radius)
			return true;
		return false;
	}
};



double circleArea(Circle circ, int maxRecLvl, int myRecLvl, Point sqrCenter) {
	if (maxRecLvl < myRecLvl) {
		return 0;
	}
	if (sqrCenter.x == sqrCenter.y)
	{
		//modify result to be *0.5 because we are on a symmetry line
		double sqrSide = sideLengthVector.at(myRecLvl);
		//TODO: filter based on symmetry
		if (circ.contains(sqrCenter))
		{
			//TODO: botleft inside, check rest
			//botleft square of next reclvl that is inside
			double sqrArea = sqrSide * sqrSide / 4;
			double sum = sqrArea / 2;
			//check topRight corner of topLeftSquare
			if (circ.contains(Point(sqrCenter.x, sqrCenter.y + (sqrSide / 2))))
			{
				sum += sqrArea; //on a symmetry line
			}
			else
			{
	#pragma omp task shared(sum) firstprivate(circ, maxRecLvl, myRecLvl, sqrCenter, sqrSide) untied
				sum += circleArea(circ, maxRecLvl, myRecLvl + 1, 
					Point(sqrCenter.x - (sqrSide / 4), sqrCenter.y + (sqrSide / 4)));
			}
			//we are at a symmetry line and will not concern ourselves with the botright square
			//split topRightSquare
	#pragma omp task shared(sum) firstprivate(circ, maxRecLvl, myRecLvl, sqrCenter, sqrSide) untied
			sum += circleArea(circ, maxRecLvl, myRecLvl + 1, 
					Point(sqrCenter.x + (sqrSide / 4), sqrCenter.y + (sqrSide / 4)));
	#pragma omp taskwait
			return sum;

		} 
		else
		{
			//TODO: topright outside, check rest
			//topRight square of next reclvl that is outside
			double sum = 0;
			//check botLeft corner of topLeftSquare
			if (circ.contains(Point(sqrCenter.x - (sqrSide / 2), sqrCenter.y)))
			{
	#pragma omp task shared(sum) firstprivate(circ, maxRecLvl, myRecLvl, sqrCenter, sqrSide) untied
				sum += circleArea(circ, maxRecLvl, myRecLvl + 1, 
					Point(sqrCenter.x - (sqrSide / 4), sqrCenter.y + (sqrSide / 4)));
			}
			//we are at a symmetry line and will not concern ourselves with the botright square
			//split botLeftSquare
	#pragma omp task shared(sum) firstprivate(circ, maxRecLvl, myRecLvl, sqrCenter, sqrSide) untied
			sum += circleArea(circ, maxRecLvl, myRecLvl + 1, 
					Point(sqrCenter.x - (sqrSide / 4), sqrCenter.y - (sqrSide / 4)));
	#pragma omp taskwait
			return sum;
		}
	}
	else if (sqrCenter.x > sqrCenter.y)
	{
		return 0;
	}
	else
	{
		double sqrSide = sideLengthVector.at(myRecLvl);
		//TODO: filter based on symmetry
		if (circ.contains(sqrCenter))
		{
			//TODO: botleft inside, check rest
			//botleft square of next reclvl that is inside
			double sqrArea = sqrSide * sqrSide / 4;
			double sum = sqrArea;
			//check topRight corner of topLeftSquare
			if (circ.contains(Point(sqrCenter.x, sqrCenter.y + (sqrSide / 2))))
			{
				sum += sqrArea;
			}
			else
			{
	#pragma omp task shared(sum) firstprivate(circ, maxRecLvl, myRecLvl, sqrCenter, sqrSide) untied
				sum += circleArea(circ, maxRecLvl, myRecLvl + 1, 
					Point(sqrCenter.x - (sqrSide / 4), sqrCenter.y + (sqrSide / 4)));
			}
			//check topRight corner of  botRightSquare
			if (circ.contains(Point(sqrCenter.x + (sqrSide / 2), sqrCenter.y)))
			{
				sum += sqrArea;
			}
			else
			{
	#pragma omp task shared(sum) firstprivate(circ, maxRecLvl, myRecLvl, sqrCenter, sqrSide) untied
				sum += circleArea(circ, maxRecLvl, myRecLvl + 1, 
					Point(sqrCenter.x + (sqrSide / 4), sqrCenter.y - (sqrSide / 4)));
			}
			//split topRightSquare
	#pragma omp task shared(sum) firstprivate(circ, maxRecLvl, myRecLvl, sqrCenter, sqrSide) untied
			sum += circleArea(circ, maxRecLvl, myRecLvl + 1, 
					Point(sqrCenter.x + (sqrSide / 4), sqrCenter.y + (sqrSide / 4)));
	#pragma omp taskwait
			return sum;

		} 
		else
		{
			//TODO: topright outside, check rest
			//topRight square of next reclvl that is outside
			double sum = 0;
			//check botLeft corner of topLeftSquare
			if (circ.contains(Point(sqrCenter.x - (sqrSide / 2), sqrCenter.y)))
			{
	#pragma omp task shared(sum) firstprivate(circ, maxRecLvl, myRecLvl, sqrCenter, sqrSide) untied
				sum += circleArea(circ, maxRecLvl, myRecLvl + 1, 
					Point(sqrCenter.x - (sqrSide / 4), sqrCenter.y + (sqrSide / 4)));
			}
			//check botLeft corner of botRightSquare
			if (circ.contains(Point(sqrCenter.x, sqrCenter.y - (sqrSide / 2))))
			{
	#pragma omp task shared(sum) firstprivate(circ, maxRecLvl, myRecLvl, sqrCenter, sqrSide) untied
				sum += circleArea(circ, maxRecLvl, myRecLvl + 1, 
					Point(sqrCenter.x + (sqrSide / 4), sqrCenter.y - (sqrSide / 4)));
			}
			//split botLeftSquare
	#pragma omp task shared(sum) firstprivate(circ, maxRecLvl, myRecLvl, sqrCenter, sqrSide) untied
			sum += circleArea(circ, maxRecLvl, myRecLvl + 1, 
					Point(sqrCenter.x - (sqrSide / 4), sqrCenter.y - (sqrSide / 4)));
	#pragma omp taskwait
			return sum;
		}
	}

	
	
}

double circleAreaBoot(Circle circ, int maxRecLvl, Point sqrCenter)	{
	return 8 * circleArea(circ, maxRecLvl, 1, sqrCenter);
}

int main()
{
	double radius;
	std::cin >> radius;
	int recLvl;
	std::cin >> recLvl;
	Circle circle;
	circle.mid = Point(0, 0);
	circle.radius = radius;
	//first sensible Center point
	Point firstCenter = Point(radius / 2, radius / 2);

	sideLengthVector.emplace_back(2 * radius);
	for (int i = 1; i <= recLvl; i++)
	{
		sideLengthVector.emplace_back(sideLengthVector[i - 1] / 2);
	}

	
	double area = 0;
	Timer timer;

	timer.start();

	//TODO: reevaluate visibility
	#pragma omp parallel shared(area, circle, recLvl, firstCenter, sideLengthVector)
	{
		#pragma omp single nowait
		{
			// work parallel magic
			area = circleAreaBoot(circle, recLvl, firstCenter);
		}
	}

	timer.stop();
	
    std::cout << area;

	return 0;
}