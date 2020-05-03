#include <GL/glut.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <eigen3/Eigen/IterativeLinearSolvers>

using namespace std;

const static int WINDOW_WIDTH = 700;
const static int WINDOW_HEIGHT = 500;

typedef Eigen::Vector2f Vec2;
typedef pair<uint, uint> Edge;
typedef vector<uint> Face;

vector<Vec2> vertices;
vector<Face> faces;
vector<Edge> contours;
vector<vector<uint>> neighboors;
vector<float> angles;
float cross_size;
size_t nV, nE;
float x_0=1e10, x_1=-1e10;
float y_0=1e10, y_1=-1e10;

bool loadOBJ(std::string filename) {
	ifstream file(filename);
	string type;
	float x, y;
	while(file >> type) {
		if(type == "v") {
			file >> x >> y;
			x_0 = min(x_0, x), x_1 = max(x_1, x);
			y_0 = min(y_0, y), y_1 = max(y_1, y);
			vertices.emplace_back(x, y);
			file >> x;
		} else if(type == "vt") {
			file >> x >> y;
		} else if(type == "f") {
			Face face;
			for(int i = 0; i < 3; i++) {
				if(!(file >> type)) return false;
				face.push_back(stoi(type.substr(0, type.find('/')))-1);
			}
			faces.push_back(face);
		} else return false;
	}
	file.close();
	nV = vertices.size();
	float w = x_1-x_0, h = y_1-y_0;
	x_0 -= 0.1*w;
	x_1 += 0.1*w;
	y_0 -= 0.1*h;
	y_1 += 0.1*h;
	return true;
}

void computeNeigbhboorsAndContours() {
	neighboors = vector<vector<uint>>(nV);
	vector<uint> lim(nV-1, 0);
	for(Face &f : faces) {
		size_t n = f.size();
		for(uint i = 0; i < n; i++) {
			uint a = f[i], b = f[(i+1)%n];
			if(a > b) swap(a, b);
			size_t nn = neighboors[a].size();
			uint j = lim[a];
			while(j < nn && neighboors[a][j] != b) j ++;
			if(j == nn) neighboors[a].push_back(b);
			else swap(neighboors[a][j], neighboors[a][lim[a]++]);
		}
	}
	for(int a = nV-2; a >= 0; a--) {
		nE += neighboors[a].size();
		for(uint b : neighboors[a])
			neighboors[b].push_back(a);
		for(uint i = lim[a]; i < neighboors[a].size(); i++)
			contours.emplace_back(a, neighboors[a][i]);
	}
}

void smooth_cross_field(uint n_steps) {
	vector<Eigen::Triplet<double>> IJV;
	size_t m = contours.size()*4 + 2*nE;
	IJV.reserve(m+2*nE);
	Eigen::VectorXd X, Y(m);
	Eigen::SparseMatrix<double> A(m, 2*nV), At(2*nV, m);
	double hard = 200.;
	uint i = 0;
	cross_size = 0;
	for(const Edge &e : contours) {
		uint a = e.first, b = e.second;
		Vec2 v = vertices[b] - vertices[a];
		cross_size += v.norm();
		double theta = 4*atan2(v[1], v[0]);
		double c = hard*cos(theta), s = hard*sin(theta);
		IJV.emplace_back(i, 2*a, hard); Y(i++) = c;
		IJV.emplace_back(i, 2*a+1, hard); Y(i++) = s;
		IJV.emplace_back(i, 2*b, hard); Y(i++) = c;
		IJV.emplace_back(i, 2*b+1, hard); Y(i++) = s;
	}
	cross_size *= .33/contours.size();
	for(uint a = 0; a < nV; ++a){
		for(uint b : neighboors[a]) if(a < b) {
			IJV.emplace_back(i, 2*a, 1.); IJV.emplace_back(i, 2*b, -1.); Y(i++) = 0.;
			IJV.emplace_back(i, 2*a+1, 1.); IJV.emplace_back(i, 2*b+1, -1.); Y(i++) = 0.;
		}
	}

	Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>> solver;
	for(uint step = 0; step < n_steps; ++step) {
		if(step > 0) {
			if(step == 1) {
				IJV.resize(IJV.size()+2*nV);
				Y.conservativeResize(m+nV);
				A.resize(m+nV, 2*nV);
			}
			double soft = 1. + (hard-1.)*min(1., 2.*step/float(n_steps));
			for(uint a = 0; a < nV; ++a) {
				double fact = soft / sqrt(X(2*a)*X(2*a) + X(2*a+1)*X(2*a+1));
				uint i = Y.size()-1-a;
				IJV[IJV.size()-1-2*a] = Eigen::Triplet<double>(i, 2*a, fact*X(2*a));
				IJV[IJV.size()-1-(2*a+1)] = Eigen::Triplet<double>(i, 2*a+1, fact*X(2*a+1));
				Y(i) = soft;
			}
		}
		A.setFromTriplets(IJV.begin(), IJV.end());
		if(step == 0) X = solver.compute(A).solve(Y);
		else{
			if(step == 1) solver.analyzePattern(A);
			X = solver.factorize(A).solveWithGuess(Y, X);
		}
	}

	angles.resize(nV);
	for(uint a = 0; a < nV; ++a) angles[a] = atan2(X(2*a+1), X(2*a)) / 4.;
}

void drawEdge(uint a, uint b) {
	glVertex2f(vertices[a][0], vertices[a][1]),
	glVertex2f(vertices[b][0], vertices[b][1]);
}
void drawEdge(const Edge &e) { drawEdge(e.first, e.second); }

void Render(void) {
	glClear(GL_COLOR_BUFFER_BIT);
	
	glLoadIdentity();
	float x = .5*(x_0+x_1), y = .5*(y_0+y_1);
	float w_rat = float(glutGet(GLUT_WINDOW_WIDTH)) / float(WINDOW_WIDTH);
	float h_rat = float(glutGet(GLUT_WINDOW_HEIGHT)) / float(WINDOW_HEIGHT);
	gluOrtho2D(x - w_rat*(x-x_0), x + w_rat*(x_1-x), y - h_rat*(y-y_0), y + h_rat*(y_1-y));

	// Draw Triangles
	glColor3f(0.2, 0.2, 0.1);
	glBegin(GL_TRIANGLES);
	for(Face &f : faces)
		for(uint i : f)
			glVertex2f(vertices[i][0], vertices[i][1]);
	glEnd();

	// Draw Edges
	glLineWidth((GLfloat) 0.4);
	glColor3f(0.1, 0.5, 0.3);
	glBegin(GL_LINES);
	for(uint a = 0; a < nV; a++) {
		for(uint b : neighboors[a]) {
			if(b < a) break;
			drawEdge(a, b);
		}
	}
	glEnd();

	// Draw Contour
	glLineWidth((GLfloat) 5.0);
	glColor3f(0.9, 0.0, 0.5);
	glBegin(GL_LINES);
	for(const Edge &e : contours) drawEdge(e);
	glEnd();

	// Draw crosses
	glLineWidth((GLfloat) 2.0);
	glColor3f(0.7, 0.7, 0.1);
	glBegin(GL_LINES);
	for(uint a = 0; a < nV; a++) {
		float co = cross_size * cos(angles[a]), si = cross_size * sin(angles[a]);
		float x = vertices[a][0], y = vertices[a][1];
		glVertex2f(x-co, y-si);
		glVertex2f(x+co, y+si);
		glVertex2f(x+si, y-co);
		glVertex2f(x-si, y+co);
	}
	glEnd();

	glutSwapBuffers();
}

float click_x, click_y;
bool clicked=false;
void Mouse(int but, int st, int x, int y) {
	float w = glutGet(GLUT_WINDOW_WIDTH), h = glutGet(GLUT_WINDOW_HEIGHT);
	float xm = .5*(x_0+x_1), ym = .5*(y_0+y_1);
	float w_rat = w / float(WINDOW_WIDTH);
	float h_rat = h / float(WINDOW_HEIGHT);
	x_0 = xm - w_rat*(xm-x_0); x_1 = xm + w_rat*(x_1-xm);
	y_0 = ym - h_rat*(ym-y_0); y_1 = ym + h_rat*(y_1-ym);
	float pw = x_1-x_0, ph = y_1-y_0;
	float px = (x * pw) / w + x_0;
	float py = y_1 - (y * ph) / h;
	bool redisplay = false;
	if(but == 3 || but == 4) {
		float fac = but == 3 ? 0.95 : 1.0/0.95;
		x_0 = px - fac*(px - x_0); x_1 = px + fac*(x_1 - px);
		y_0 = py - fac*(py - y_0); y_1 = py + fac*(y_1 - py);
		redisplay = true;
	} else if(but == 0) {
		if(st == GLUT_DOWN) { click_x = px; click_y = py; clicked = true; }
		else clicked = false;
	}
	xm = .5*(x_0+x_1); ym = .5*(y_0+y_1);
	x_0 = xm - (xm-x_0)/w_rat; x_1 = xm + (x_1-xm)/w_rat;
	y_0 = ym - (ym-y_0)/h_rat; y_1 = ym + (y_1-ym)/h_rat;
	if(redisplay) glutPostRedisplay();
}

void Move(int x, int y) {
	if(!clicked) return;
	float w = glutGet(GLUT_WINDOW_WIDTH), h = glutGet(GLUT_WINDOW_HEIGHT);
	float xm = .5*(x_0+x_1), ym = .5*(y_0+y_1);
	float w_rat = w / float(WINDOW_WIDTH);
	float h_rat = h / float(WINDOW_HEIGHT);
	x_0 = xm - w_rat*(xm-x_0); x_1 = xm + w_rat*(x_1-xm);
	y_0 = ym - h_rat*(ym-y_0); y_1 = ym + h_rat*(y_1-ym);
	float pw = x_1-x_0, ph = y_1-y_0;
	float dx = click_x - ((x * pw) / w + x_0);
	float dy = click_y - (y_1 - (y * ph) / h);
	x_0 = xm+dx - (xm-x_0)/w_rat; x_1 = xm+dx + (x_1-xm)/w_rat;
	y_0 = ym+dy - (ym-y_0)/h_rat; y_1 = ym+dy + (y_1-ym)/h_rat;
	glutPostRedisplay();
}

int main(int argc, char** argv) {
	if(argc != 2) {
		cerr << "Exactly one argument needed. This argument is the path to an OBJ file." << std::endl;
		return 1;
	}
	if(!loadOBJ(argv[1]))  {
		cerr << "Fail to load the OBJ file !" << std::endl;
		return 1;
	}
	computeNeigbhboorsAndContours();
	smooth_cross_field(15);

	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
	glutInit(&argc, argv);
	glutCreateWindow("Quads");
	glClearColor(0.9f,0.9f,0.9f,1);
	glEnable(GL_LINE_SMOOTH);

	glutDisplayFunc(Render);
	glutMouseFunc(Mouse);
	glutMotionFunc(Move);

	glutMainLoop();
	return 0;
}