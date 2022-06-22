#include "DenoiseSystem.h"

#include "../Components/DenoiseData.h"

#include <_deps/imgui/imgui.h>

#include <spdlog/spdlog.h>
#include"Eigen/Core"
#include"Eigen/Sparse"

using namespace Ubpa;

void DenoiseSystem::OnUpdate(Ubpa::UECS::Schedule& schedule) {
	schedule.RegisterCommand([](Ubpa::UECS::World* w) {
		auto data = w->entityMngr.GetSingleton<DenoiseData>();
		if (!data)
			return;

		if (ImGui::Begin("Denoise")) {
			if (ImGui::Button("Mesh to HEMesh")) {
				data->heMesh->Clear();
				[&]() {
					if (!data->mesh) {
						spdlog::warn("mesh is nullptr");
						return;
					}

					if (data->mesh->GetSubMeshes().size() != 1) {
						spdlog::warn("number of submeshes isn't 1");
						return;
					}

					data->copy = *data->mesh;

					std::vector<size_t> indices(data->mesh->GetIndices().begin(), data->mesh->GetIndices().end());
					data->heMesh->Init(indices, 3);
					if (!data->heMesh->IsTriMesh())
						spdlog::warn("HEMesh init fail");
					
					for (size_t i = 0; i < data->mesh->GetPositions().size(); i++) {
						data->heMesh->Vertices().at(i)->position = data->mesh->GetPositions().at(i);
						data->heMesh->Vertices().at(i)->normal = data->mesh->GetNormals().at(i);
						data->heMesh->Vertices().at(i)->index = -1;
						data->heMesh->Vertices().at(i)->Bound_index = -1;
					}

					spdlog::info("Mesh to HEMesh success");
				}();
			}

			if (ImGui::Button("Global Laplacian Smoothing")) {
				[&]() {
					if (!data->heMesh->IsTriMesh()) {
						spdlog::warn("HEMesh isn't triangle mesh");
						return;
					}

					const auto vertices = data->heMesh->Vertices();//��ȡ����
					int V_num = vertices.size();//��������
					int V_OnBoundary = 0;//�߽����
					int V_Inner = 0;//�Ǳ߽����
					for (int i = 0; i < V_num; i++) {
						//������������
						vertices[i]->index = i;
						if (vertices[i]->IsOnBoundary()) {
							//�������λ�ڱ߽�㣬���ӱ߽����
							V_OnBoundary++;
						}
					}
					V_Inner = V_num - V_OnBoundary;//�Ǳ߽����

					//���ȫ�ּ�С�������
					Eigen::SparseMatrix <float> A(3 * (V_num + V_OnBoundary), 3 * V_num);
					Eigen::SparseVector<float> b(3 * (V_num + V_OnBoundary));

					//�����������
					spdlog::info(V_num);

					//��������
					auto Idx = [&](int idx, int k) {
						return k * V_num + idx;
					};
					auto B_Idx = [&](int idx, int k) {
						return 3 * V_num + idx * 3 + k;
					};

					int Boundary_index = 0;
					for (auto* v : data->heMesh->Vertices()) {
						//�������ж��㣬����ϡ�����
						const auto Pos = v->position;//��ȡ����λ��

						//A_ij & x_i
						if (v->IsOnBoundary()) {
							//����Ǳ߽綥��
							//�洢1 or 0��ʾ�Ƿ�Ϊ�߽綥��
							for (auto k : { 0,1,2 }) {
								A.coeffRef(B_Idx(Boundary_index, k), Idx(v->index, k)) = 1;
								b.coeffRef(B_Idx(Boundary_index, k)) = Pos[k];
							}
							Boundary_index++;
						}
						//�Ե�ǰ����1-���򶥵�
						const auto& adj_v = v->AdjVertices();
						for (auto k : { 0, 1, 2 }) {
							//�Զ�������
							//�洢1-���򶥵������
							A.coeffRef(Idx(v->index, k), Idx(v->index, k)) = v->AdjVertices().size();
						}

						for (auto* adj : adj_v) {
							//�Ե�ǰ����� 1-���򶥵�����
							//�ǵ�ǰ����� 1-���򶥵���Ϊ-1
							for (auto k : { 0, 1, 2 }) {
								A.coeffRef(Idx(v->index, k), Idx(adj->index, k)) = -1;
							}
						}
					}
					//���󹹽��ɹ�
					spdlog::info("Build Matrix successful");
					//������
					Eigen::ConjugateGradient<Eigen::SparseMatrix<float> > solver;
					solver.setTolerance(1e-4);
					solver.compute(A.transpose()* A);
					Eigen::SparseVector<float> x(3 * V_num);
					x = solver.solve(A.transpose() * b);

					//������
					spdlog::info("Solve Matrix successful");

					//�������ж�������
					for (auto* v : data->heMesh->Vertices()) {
						for (auto k : { 0,1,2 }) {
							v->position[k] = x.coeffRef(Idx(v->index, k));
						}
					}
					spdlog::info("Data transform successful");
				}();
			}

			//if (ImGui::Button("Add Noise")) {
			//	[&]() {
			//		if (!data->heMesh->IsTriMesh()) {
			//			spdlog::warn("HEMesh isn't triangle mesh");
			//			return;
			//		}

			//		for (auto* v : data->heMesh->Vertices()) {
			//			v->position += data->randomScale * (
			//				2.f * Ubpa::vecf3{ Ubpa::rand01<float>(),Ubpa::rand01<float>() ,Ubpa::rand01<float>() } - Ubpa::vecf3{ 1.f }
			//			);
			//		}

			//		spdlog::info("Add noise success");
			//	}();
			//}


			if (ImGui::Button("Parameterization")) {
				[&]() {
					if (!data->heMesh->IsTriMesh()) {
						spdlog::warn("HEMesh isn't triangle mesh");
						return;
					}

					const auto vertices = data->heMesh->Vertices();//��ȡ����
					int V_num = vertices.size();//��������
					int V_OnBoundary = 0;//�߽����
					int V_Inner = 0;//�Ǳ߽����
					for (int i = 0; i < V_num; i++) {
						//������������
						vertices[i]->index = i;
						if (vertices[i]->IsOnBoundary()) {
							if (vertices[i]->Bound_index < 0) {
								//�����߽������
								vertices[i]->Bound_index = V_OnBoundary;
							}
							//�������λ�ڱ߽�㣬���ӱ߽����
							V_OnBoundary++;
						}
					}

					//for (int i = 0; i < V_num; ++i) vertices[i]->index = i;
					//int bpt;
					//for (int i = 0; i < V_num; ++i)
					//	if (vertices[i]->IsOnBoundary())
					//		bpt = vertices[i]->index; // start point

					//do {
					//	vertices[bpt]->Bound_index = V_OnBoundary++;
					//	const auto& adj_v = vertices[bpt]->AdjVertices();
					//	for (auto* v : data->heMesh->Vertices()) {
					//		if (v->IsOnBoundary() && v->Bound_index < 0) {
					//			bpt = v->index;
					//			break;
					//		}
					//	}
					//} while (vertices[bpt]->Bound_index < 0);

					V_Inner = V_num - V_OnBoundary;//�Ǳ߽����

					//�����������
					spdlog::info(V_num);

					//��������
					auto Idx = [&](int idx, int k) {
						return k * V_num + idx;
					};
					auto B_Idx = [&](int idx, int k) {
						return 2 * V_num + idx * 2 + k;
					};

					//������������
					Eigen::SparseMatrix<float> A(2 * (V_num + V_OnBoundary), 2 * V_num);
					Eigen::SparseVector<float> b(2 * (V_num + V_OnBoundary));

					for (auto* v : data->heMesh->Vertices()) {
						const auto Pos = v->position;

						if (v->IsOnBoundary()) {
							{//����Ǳ߽綥��
							//���߽綥��ӳ�䵽uv����߽���[0,1]^2
								float t = 4 * float(v->Bound_index) / float(V_OnBoundary); //����t��Ϊ�ı�
								float U = { 0.0f }, V = { 0.0f };//��ʼ��uv����
								if (t <= 1.0f) V = t;
								else if (t <= 2.0f) U = t - 1, V = 1.0f;
								else if (t <= 3.0f) U = 1.0f, V = 3.0f - t;
								else U = 4.0f - t;

								//���UV����
								spdlog::info(std::to_string(U) + " " + std::to_string(V));

								//uv��������ֵ
								b.coeffRef(B_Idx(v->Bound_index, 0)) = U;
								b.coeffRef(B_Idx(v->Bound_index, 1)) = V;
							}
							for (auto k : { 0, 1 }) 
								//����ֵ
								A.coeffRef(B_Idx(v->Bound_index , k), Idx(v->index, k)) = 1;
						}
						const auto& adj_v = v->AdjVertices();
						for (auto k : { 0, 1 })
							A.coeffRef(Idx(v->index, k), Idx(v->index, k)) = v->AdjVertices().size();

						for (auto* adj : adj_v) {
							for (auto k : { 0, 1 }) A.coeffRef(Idx(v->index, k), Idx(adj->index, k)) = -1;
						}
					}
					spdlog::info("Build Matrix successful");

					Eigen::ConjugateGradient<Eigen::SparseMatrix<float> > solver;
					solver.setTolerance(1e-2);
					solver.compute(A.transpose()* A);
					Eigen::SparseVector<float> x(2 * V_num);
					x = solver.solve(A.transpose() * b);

					spdlog::info("Solve Matrix successful");

					for (auto* v : data->heMesh->Vertices()) {
						v->UV = { x.coeffRef(Idx(v->index, 0)), x.coeffRef(Idx(v->index, 1)) };
						spdlog::info(std::to_string(v->UV[0]) + " " + std::to_string(v->UV[1]));
					}

					spdlog::info("Data transform successful");

				}();
			}


			if (ImGui::Button("Set Normal to Color")) {
				[&]() {
					if (!data->mesh) {
						spdlog::warn("mesh is nullptr");
						return;
					}

					data->mesh->SetToEditable();
					const auto& normals = data->mesh->GetNormals();
					std::vector<rgbf> colors;
					for (const auto& n : normals)
						colors.push_back((n.as<valf3>() + valf3{ 1.f }) / 2.f);
					data->mesh->SetColors(std::move(colors));

					spdlog::info("Set Normal to Color Success");
				}();
			}

			if (ImGui::Button("HEMesh to Mesh")) {
				[&]() {
					if (!data->mesh) {
						spdlog::warn("mesh is nullptr");
						return;
					}

					if (!data->heMesh->IsTriMesh() || data->heMesh->IsEmpty()) {
						spdlog::warn("HEMesh isn't triangle mesh or is empty");
						return;
					}

					data->mesh->SetToEditable();

					const size_t N = data->heMesh->Vertices().size();
					const size_t M = data->heMesh->Polygons().size();
					std::vector<Ubpa::pointf3> positions(N);
					std::vector<Ubpa::pointf2> UVs(N);
					std::vector<uint32_t> indices(M * 3);
					for (size_t i = 0; i < N; i++) {
						positions[i] = data->heMesh->Vertices().at(i)->position;
						UVs[i] = data->heMesh->Vertices().at(i)->UV;
					}

					for (size_t i = 0; i < M; i++) {
						auto tri = data->heMesh->Indices(data->heMesh->Polygons().at(i));
						indices[3 * i + 0] = static_cast<uint32_t>(tri[0]);
						indices[3 * i + 1] = static_cast<uint32_t>(tri[1]);
						indices[3 * i + 2] = static_cast<uint32_t>(tri[2]);
					}

					data->mesh->SetPositions(std::move(positions));
					data->mesh->SetUV(std::move(UVs));
					data->mesh->SetIndices(std::move(indices));
					data->mesh->SetColors({});
					data->mesh->SetSubMeshCount(1);
					data->mesh->SetSubMesh(0, { 0, M * 3 });
					//data->mesh->GenUV();
					data->mesh->GenNormals();
					data->mesh->GenTangents();

					spdlog::info("HEMesh to Mesh success");
				}();
			}

			if (ImGui::Button("Recover Mesh")) {
				[&]() {
					if (!data->mesh) {
						spdlog::warn("mesh is nullptr");
						return;
					}
					if (data->copy.GetPositions().empty()) {
						spdlog::warn("copied mesh is empty");
						return;
					}

					*data->mesh = data->copy;

					spdlog::info("recover success");
				}();
			}
		}
		ImGui::End();
	});
}
