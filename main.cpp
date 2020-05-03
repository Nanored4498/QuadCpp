#include <GL/glut.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <random>

using namespace std;

const static int WINDOW_WIDTH = 700;
const static int WINDOW_HEIGHT = 500;

default_random_engine rand_generator;

struct Vertex {
	float x, y;
	Vertex(float x, float y): x(x), y(y) {}
	Vertex operator-(const Vertex &other) { return Vertex(x-other.x, y-other.y); }
	float norm() { return sqrt(x*x + y*y); }
};
typedef pair<uint, uint> Edge;
typedef vector<uint> Face;

vector<Vertex> vertices;
vector<Face> faces;
vector<Edge> contours;
vector<vector<uint>> neighboors;
vector<float> angles;
uint ring_vert = 987654321;
float cross_size;
size_t nV;
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
		for(uint b : neighboors[a])
			neighboors[b].push_back(a);
		for(uint i = lim[a]; i < neighboors[a].size(); i++)
			contours.emplace_back(a, neighboors[a][i]);
	}
}

void rand_angles() {
	uniform_int_distribution<uint> random_vertex(0, vertices.size()-1);
	ring_vert = random_vertex(rand_generator);
	angles = vector<float>(nV);
	uniform_real_distribution<float> random_angle(0, 0.5*M_PI);
	for(uint i = 0; i < nV; i++) angles[i] = random_angle(rand_generator);
	cross_size = 0;
	for(Edge &e : contours) angles[e.first] = 0;
	for(Edge &e : contours) {
		uint a = e.first, b = e.second;
		Vertex v = vertices[b] - vertices[a];
		float add = 0.5*atan2(v.y, v.x);
		angles[a] += add, angles[b] += add;
		cross_size += v.norm();
	}
	cross_size *= 0.33/contours.size();
}

void smooth_cross_field(uint n_steps) {
	// Initialize
	angles = vector<float>(nV);
	vector<bool> is_edge(nV, false);
	for(uint i = 0; i < nV; i++) angles[i] = 0;
	for(Edge &e : contours) {
		uint a = e.first, b = e.second;
		is_edge[a] = is_edge[b] = true;
		Vertex v = vertices[b] - vertices[a];
		float add = 0.5*atan2(v.y, v.x);
		angles[a] += add, angles[b] += add;
		cross_size += v.norm();
	}
	cross_size *= 0.33/contours.size();

	// Upddate
	vector<float> tmp(nV);
	for(uint step = 0; step < n_steps; step++) {
		for(uint a = 0; a < nV; a++) {
			if(is_edge[a]) tmp[a] = is_edge[a];
			else {
				float co = 0, si = 0;
				for(uint b : neighboors[a])
					co += cos(4*angles[b]), si += sin(4*angles[b]);
				tmp[a] = atan2(si, co) / 4;
			}
		}
		swap(angles, tmp);
	}
}

void drawEdge(uint a, uint b) {
	glVertex2f(vertices[a].x, vertices[a].y),
	glVertex2f(vertices[b].x, vertices[b].y);
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
			glVertex2f(vertices[i].x, vertices[i].y);
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

	// Draw ring
	if(ring_vert < nV) {
		glLineWidth((GLfloat) 3.0);
		glColor3f(0.9, 0.9, 0.0);
		glBegin(GL_LINES);
		for(uint b : neighboors[ring_vert]) drawEdge(ring_vert, b);
		glEnd();
	}

	// Draw crosses
	glLineWidth((GLfloat) 2.0);
	glColor3f(0.7, 0.7, 0.1);
	glBegin(GL_LINES);
	for(uint a = 0; a < nV; a++) {
		float co = cross_size * cos(angles[a]), si = cross_size * sin(angles[a]);
		float x = vertices[a].x, y = vertices[a].y;
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
	// rand_angles();
	smooth_cross_field(200);

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