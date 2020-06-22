#pragma once
#include <array>
#include <vector>
#include <map>
#include <list>
#include <iostream>
namespace FussenAlgo
{
	template<class TVert,class ftype>
	class MarchingSquarsCPU
	{
	public:
		//using ftype = float;
		//using TVert = std::array<ftype, 3>;
	protected:
		struct Edge {
			ftype ts[3];
			ftype te[3];
			 union {
				struct {
					int sid;
					int eid;
				};
				struct {
					int id[2];
				};
			};
		};
	public:

		void mscpuRoubst(std::vector<TVert>& pts, std::vector<ftype>& values, int udim, int vdim, ftype isoFace,
			std::vector<std::vector<TVert>>& curve, std::vector<bool>* property = nullptr)
		{
			std::vector<Edge> edges;
			edges.clear();
			int cudim = udim - 1;
			int cvdim = vdim - 1;
			edges.reserve(cudim*cvdim);
			for (int u = 0; u < cudim; ++u) {
				for (int v = 0; v < cvdim; ++v) {
					ftype v0 = values[idx(u, v, udim)];
					ftype v1 = values[idx(u + 1, v, udim)];
					ftype v2 = values[idx(u + 1, v + 1, udim)];
					ftype v3 = values[idx(u, v + 1, udim)];
					ftype p0[3], p1[3], p2[3], p3[3];
					p0[0] = pts[idx(u, v, udim)][0]; p1[0] = pts[idx(u + 1, v, udim)][0]; p2[0] = pts[idx(u + 1, v + 1, udim)][0]; p3[0] = pts[idx(u, v + 1, udim)][0];
					p0[1] = pts[idx(u, v, udim)][1]; p1[1] = pts[idx(u + 1, v, udim)][1]; p2[1] = pts[idx(u + 1, v + 1, udim)][1]; p3[1] = pts[idx(u, v + 1, udim)][1];
					p0[2] = pts[idx(u, v, udim)][2]; p1[2] = pts[idx(u + 1, v, udim)][2]; p2[2] = pts[idx(u + 1, v + 1, udim)][2]; p3[2] = pts[idx(u, v + 1, udim)][2];
					MachingKernel(v0, isoFace, v1, v2, v3, p0, p1, p3, edges, p2);
				}
			}
			sortEdges(curve, edges, property);
		}

		void mscpuRoubst(TVert*pts, ftype* values, int udim, int vdim, ftype isoFace,
			std::vector<std::vector<TVert>>& curve, std::vector<bool>* property = nullptr)
		{
			std::vector<Edge> edges;
			edges.clear();
			int cudim = udim - 1;
			int cvdim = vdim - 1;
			edges.reserve(cudim*cvdim);
			for (int u = 0; u < cudim; ++u) {
				for (int v = 0; v < cvdim; ++v) {
					ftype v0 = values[idx(u, v, udim)];
					ftype v1 = values[idx(u+1, v, udim)];
					ftype v2 = values[idx(u+1, v+1, udim)];
					ftype v3 = values[idx(u, v + 1, udim)];
					ftype p0[3], p1[3], p2[3], p3[3];
					p0[0] = pts[idx(u,v,udim)][0]; p1[0] = pts[idx(u+1,v,udim)][0]; p2[0] = pts[idx(u+1,v+1,udim)][0]; p3[0] = pts[idx(u,v+1,udim)][0];
					p0[1] = pts[idx(u,v,udim)][1]; p1[1] = pts[idx(u+1,v,udim)][1]; p2[1] = pts[idx(u+1,v+1,udim)][1]; p3[1] = pts[idx(u,v+1,udim)][1];
					p0[2] = pts[idx(u,v,udim)][2]; p1[2] = pts[idx(u+1,v,udim)][2]; p2[2] = pts[idx(u+1,v+1,udim)][2]; p3[2] = pts[idx(u,v+1,udim)][2];
					MachingKernel(v0, isoFace, v1, v2, v3, p0, p1, p3, edges, p2);
				}
			}
			sortEdges(curve, edges, property);
		}

		void mscpuRoubst(TVert** pts, ftype** values, int udim, int vdim,
			ftype isoFace, std::vector<std::vector<TVert>>& curve,std::vector<bool>* property=nullptr)
		{
			std::vector<Edge> edges;
			edges.clear();
			int cudim = udim - 1;
			int cvdim = vdim - 1;
			edges.reserve(cudim*cvdim);
			for (int u = 0; u < cudim; ++u) {
				for (int v = 0; v < cvdim; ++v) {
					ftype v0 = values[u][v];
					ftype v1 = values[u + 1][v];
					ftype v2 = values[u + 1][v + 1];
					ftype v3 = values[u][v + 1];
					ftype p0[3], p1[3], p2[3], p3[3];
					p0[0] = pts[u][v][0]; p1[0] = pts[u + 1][v][0]; p2[0] = pts[u + 1][v + 1][0]; p3[0] = pts[u][v + 1][0];
					p0[1] = pts[u][v][1]; p1[1] = pts[u + 1][v][1]; p2[1] = pts[u + 1][v + 1][1]; p3[1] = pts[u][v + 1][1];
					p0[2] = pts[u][v][2]; p1[2] = pts[u + 1][v][2]; p2[2] = pts[u + 1][v + 1][2]; p3[2] = pts[u][v + 1][2];
					MachingKernel(v0, isoFace, v1, v2, v3, p0, p1, p3, edges, p2);
				}
			}
			sortEdges(curve, edges,property);
		}
	protected:
		inline void MachingKernel(ftype &v0, ftype &isoFace, ftype &v1, ftype &v2, ftype &v3, ftype  p0[3], ftype  p1[3], ftype  p3[3], 
			std::vector<Edge> &edges, ftype  p2[3])
		{
			int config =
				((v0 <= isoFace) << 0) |
				((v1 <= isoFace) << 1) |
				((v2 <= isoFace) << 2) |
				((v3 <= isoFace) << 3);
			switch (config)
			{
			case 0x0:
			case 0xf:
				break;
			case 0x1:
			case 0xe:
			{
				Edge edge;
				msEdgeS(edge, p0, p1, v0, v1, isoFace);
				msEdgeE(edge, p0, p3, v0, v3, isoFace);
				edges.emplace_back(edge);
			}
			break;
			case 0x2:
			case 0xd:
			{
				Edge edge;
				msEdgeS(edge, p0, p1, v0, v1, isoFace);
				msEdgeE(edge, p1, p2, v1, v2, isoFace);
				edges.emplace_back(edge);
			}
			break;
			case 0x4:
			case 0xb:
			{
				Edge edge;
				msEdgeS(edge, p1, p2, v1, v2, isoFace);
				msEdgeE(edge, p3, p2, v3, v2, isoFace);
				edges.emplace_back(edge);
			}
			break;
			case 0x8:
			case 0x7:
			{
				Edge edge;
				msEdgeS(edge, p0, p3, v0, v3, isoFace);
				msEdgeE(edge, p3, p2, v3, v2, isoFace);
				edges.emplace_back(edge);
			}
			break;
			case 0xc:
			case 0x3:
			{
				Edge edge;
				msEdgeS(edge, p0, p3, v0, v3, isoFace);
				msEdgeE(edge, p1, p2, v1, v2, isoFace);
				edges.emplace_back(edge);
			}
			break;
			case 0x6:
			case 0x9:
			{
				Edge edge;
				msEdgeS(edge, p0, p1, v0, v1, isoFace);
				msEdgeE(edge, p3, p2, v3, v2, isoFace);
				edges.emplace_back(edge);
			}
			break;
			case 0x5:
			{
				ftype cenv = (v0 + v1 + v2 + v3)*0.25f;
				if (cenv >= isoFace) {
					Edge edge;
					msEdgeS(edge, p0, p3, v0, v3, isoFace);
					msEdgeE(edge, p3, p2, v3, v2, isoFace);
					edges.emplace_back(edge);
					msEdgeS(edge, p0, p1, v0, v1, isoFace);
					msEdgeE(edge, p1, p2, v1, v2, isoFace);
					edges.emplace_back(edge);
				}
				else if (cenv < isoFace) {
					Edge edge;
					msEdgeS(edge, p0, p1, v0, v1, isoFace);
					msEdgeE(edge, p0, p3, v0, v3, isoFace);
					edges.emplace_back(edge);
					msEdgeS(edge, p1, p2, v1, v2, isoFace);
					msEdgeE(edge, p3, p2, v3, v2, isoFace);
					edges.emplace_back(edge);
				}
			}
			break;
			case 0xa:
			{
				ftype cenv = (v0 + v1 + v2 + v3)*0.25f;
				if (cenv >= isoFace) {
					Edge edge;
					msEdgeS(edge, p0, p1, v0, v1, isoFace);
					msEdgeE(edge, p0, p3, v0, v3, isoFace);
					edges.emplace_back(edge);
					msEdgeS(edge, p1, p2, v1, v2, isoFace);
					msEdgeE(edge, p3, p2, v3, v2, isoFace);
					edges.emplace_back(edge);
				}
				else if (cenv < isoFace) {
					Edge edge;
					msEdgeS(edge, p0, p3, v0, v3, isoFace);
					msEdgeE(edge, p3, p2, v3, v2, isoFace);
					edges.emplace_back(edge);
					msEdgeS(edge, p0, p1, v0, v1, isoFace);
					msEdgeE(edge, p1, p2, v1, v2, isoFace);
					edges.emplace_back(edge);
				}
			}
			break;
			default:
				break;
			}
		}

	public:
		template<class T>
		void allocator(T**& data, int udim, int vdim) {
			data = new T*[udim];
			for (int i = 0; i < udim; ++i)
				data[i] = new T[vdim];
		}
		template<class T>
		void garbage(T**& data, int udim, int vdim) {
			for (int i = 0; i < udim; ++i)
				delete[] data[i];
			delete[] data;
		}
	protected:
		inline void interploate(ftype p[3], ftype sp[3], ftype se[3], ftype vs, ftype ve, ftype iso) {
			ftype eps = std::numeric_limits<ftype>::epsilon();
			if ((iso - vs)*(iso - vs) < eps || (vs - ve)*(vs - ve) < eps)
			{
				p[0] = sp[0];
				p[1] = sp[1];
				p[2] = sp[2];
				return;
			}
			else if ((iso - ve)*(iso - ve) < eps) {
				p[0] = se[0];
				p[1] = se[1];
				p[2] = se[2];
				return;
			}
			else {
				ftype dp = (iso - vs) / (ve - vs);
				p[0] = (se[0] - sp[0])*dp + sp[0];
				p[1] = (se[1] - sp[1])*dp + sp[1];
				p[2] = (se[2] - sp[2])*dp + sp[2];
				return;
			}
		}
		inline void msEdgeS(Edge& edges, ftype p0[3], ftype p1[3], ftype v0, ftype v1, ftype iso)
		{
			interploate(edges.ts, p0, p1, v0, v1, iso);
		}
		inline void msEdgeE(Edge& edges, ftype p0[3], ftype p1[3], ftype v0, ftype v1, ftype iso)
		{
			interploate(edges.te, p0, p1, v0, v1, iso);
		}
		inline int idx(int u, int v, int udim) {
			return v * udim + u;
		}

	protected:
		template <typename number_t, typename index_t>
		struct CoordWithIndex {
			number_t data[3];
			index_t index;
			index_t rindex;
			bool operator == (const CoordWithIndex& c) const {
				return (c[0] == data[0]) && (c[1] == data[1]) && (c[2] == data[2]);
			}
			bool operator != (const CoordWithIndex& c) const {
				return (c[0] != data[0]) || (c[1] != data[1]) || (c[2] != data[2]);
			}
			bool operator < (const CoordWithIndex& c) const {
				return		(data[0] < c[0])
					|| (data[0] == c[0] && data[1] < c[1])
					|| (data[0] == c[0] && data[1] == c[1] && data[2] < c[2]);
			}
			inline number_t& operator [] (const size_t i) { return data[i]; }
			inline number_t operator [] (const size_t i) const { return data[i]; }
		};

	
		void sortEdges(std::vector<std::vector<TVert>>& curve, std::vector<Edge>& edges,std::vector<bool>* property) {
			using Coord = CoordWithIndex<ftype, int>;
			int nid = -1;
			std::vector<Coord> cpts;
			int npts = edges.size() * 2;
			//先对点进行Hash编码
			cpts.resize(npts);
			if (npts <= 0)
				return;
			for (auto& ie : edges) {
				++nid;
				cpts[nid][0] = ie.ts[0];
				cpts[nid][1] = ie.ts[1];
				cpts[nid][2] = ie.ts[2];
				cpts[nid].index = nid;
				++nid;
				cpts[nid][0] = ie.te[0];
				cpts[nid][1] = ie.te[1];
				cpts[nid][2] = ie.te[2];
				cpts[nid].index = nid;
			}
			std::sort(cpts.begin(), cpts.end());
			int unique = 0;

			//curve.resize(1);

			cpts[0].rindex = 0;
			TVert tp;
			tp[0] = cpts[0][0];
			tp[1] = cpts[0][1];
			tp[2] = cpts[0][2];
			curve[0].emplace_back(tp);
			for (int i = 1; i < npts; ++i) {
				if (cpts[i - 1] != cpts[i]) {
					++unique;
					tp[0] = cpts[i][0];
					tp[1] = cpts[i][1];
					tp[2] = cpts[i][2];
					curve[0].emplace_back(tp);
				}	
				cpts[i].rindex = unique;
			}
		}
	};

}
