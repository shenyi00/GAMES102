#include "CanvasSystem.h"

#include "../Components/CanvasData.h"

#include <_deps/imgui/imgui.h>
#include <eigen/Eigen/Dense>

using namespace Ubpa;

#define dataScale 10.0f
#define sig 5.0
#define pDim 4
#define lamda 2
namespace
{
	void FuncPolynomial(float x, Eigen::VectorXf& span)
	{
		int dim =(int) span.size();
		for (int i = 0; i < dim; i++) {
			span(i) = std::pow(x,  i);
		}
	}
	void ConstructSpanInsertPolynomial(const std::vector<pointf2>& input,size_t dim,Eigen::MatrixXf &m_A)
	{
		for (int i = 0; i < dim; i++) {
			for (int j = 0; j < dim; j++) {
				m_A(i, j) = std::pow(input[i][0], j);
			}
		}
	}
	void ConstructSpanApproachPolynomial(const std::vector<pointf2>& input, size_t dim, Eigen::MatrixXf& m_A,Eigen::VectorXf &v_b)
	{		
		size_t pointNum = input.size();
		m_A.resize(dim, dim);
		m_A.setZero();
		v_b.setZero();
		for (int i = 0; i < dim; i++) {
			for (int j = 0; j < dim; j++) {
				for (int k = 0; k < pointNum; k++) {
					m_A(i, j) += std::pow(input[k][0], i + j);
				}
			}
		}
		for (int i = 0; i < dim; ++i) {
			for (int k = 0; k < pointNum; ++k) {
				v_b(i) += input[k][1] * std::pow(input[k][0], i);
			}
		}
	}

	void AddRestrain(Eigen::MatrixXf& m_A)
	{
		//lamda
		size_t row = m_A.rows();
		Eigen::MatrixXf mIdenty(row, row);
		mIdenty.setIdentity();
		m_A = m_A + (lamda / dataScale) * mIdenty;
	}


	void FuncGause(float x, size_t dim, float sigma,const std::vector<pointf2>& input, Eigen::VectorXf& span)
	{
		span(0) = 1;
		for (int i = 1; i < dim; i++) {
			span(i) = std::exp(-std::pow((x - input[i][0]), 2) / 2.0f / sigma / sigma);
		}
	}
	void ConstructSpanInsertGause(const std::vector<pointf2>& input, size_t dim, float sigma,Eigen::MatrixXf& m_A)
	{
		for (int i = 0; i < dim; i++) {
			m_A(i, 0) = 1;
			for (int j = 1; j < dim; j++) {
				m_A(i, j) = std::exp(-std::pow((input[i][0] - input[j][0]), 2) / 2.0f / sigma / sigma);
			}
		}
	}

	void FitInputDataConnect(const std::vector<pointf2>& input, std::vector<pointf2>& output)
	{
		output.clear();
		for (int n = 0; n < input.size() - 1; ++n) {
			output.push_back(input[n]);
			output.push_back(input[n + 1]);
		}
	}

	void FitInputDataApproach(const std::vector<pointf2>& input,size_t spanSize,bool notAddRestrain,std::vector<pointf2>& output)
	{
		size_t pointNum = input.size();
		Eigen::MatrixXf m_A(spanSize, spanSize);
		Eigen::VectorXf v_x(spanSize);
		Eigen::VectorXf v_b(spanSize);
		ConstructSpanApproachPolynomial(input, spanSize, m_A,v_b);
		if (!notAddRestrain)
		{
			AddRestrain(m_A);
		}

		v_x = m_A.lu().solve(v_b);
		//v_x = m_A.fullPivHouseholderQr().solve(v_b);
		float coorX = input[0][0] - 5.0f;
		float maxCoorX = input[pointNum - 1][0] + 5.0f;
		do {
			float coorY = 0;
			Eigen::VectorXf v_span(spanSize);
			FuncPolynomial(coorX, v_span);
			//FuncGause(coorX, spanSize, sig, input, v_span);
			coorY = v_span.dot(v_x);

			output.push_back(pointf2(coorX, coorY));
			coorX += 2.0f / dataScale;
		} while (coorX < maxCoorX);
	}

	void FitInputDataInsert(const std::vector<pointf2>& input,bool choice ,std::vector<pointf2>& output)
	{
		size_t pointNum = input.size();
		Eigen::MatrixXf m_A(pointNum, pointNum);
		Eigen::VectorXf v_x(pointNum);
		Eigen::VectorXf v_b(pointNum);
		if(choice)
			ConstructSpanInsertPolynomial(input, pointNum, m_A);
		else
			ConstructSpanInsertGause(input, pointNum, sig, m_A);
		for (int i = 0; i < pointNum; i++) {
			v_b(i) = input[i][1];
		}
		//v_x = m_A.fullPivHouseholderQr().solve(v_b);
		v_x = m_A.lu().solve(v_b);
		float coorX = input[0][0]-5.0f;
		float maxCoorX = input[pointNum -1][0] + 5.0f;
		do{
			float coorY = 0;
			Eigen::VectorXf v_span(pointNum);
			if(choice)
				FuncPolynomial(coorX, v_span);
			else
				FuncGause(coorX, pointNum, sig, input, v_span);
			coorY = v_span.dot(v_x);

			output.push_back(pointf2(coorX, coorY));
			coorX += 2.0f/ dataScale;
		} while (coorX < maxCoorX);		
	}
}

void CanvasSystem::OnUpdate(Ubpa::UECS::Schedule& schedule) {
	schedule.RegisterCommand([](Ubpa::UECS::World* w) {
		auto data = w->entityMngr.GetSingleton<CanvasData>();
		if (!data)
			return;		
		if (ImGui::Begin("Canvas")) {
			ImGui::Checkbox("Enable grid", &data->opt_enable_grid);
			ImGui::Checkbox("Enable context menu", &data->opt_enable_context_menu);
			if(data->opt_enable_insert)
				ImGui::Checkbox("Insert", &data->opt_enable_insert);
			else
				ImGui::Checkbox("Approach", &data->opt_enable_insert);
			if (data->opt_enable_insert)
			{
				if (data->opt_choice_0_or_1)
					ImGui::Checkbox("Polynomial", &data->opt_choice_0_or_1);
				else
					ImGui::Checkbox("Gause", &data->opt_choice_0_or_1);
			}				
			else
			{
				if (data->opt_choice_0_or_1)
					ImGui::Checkbox("LeastSquare", &data->opt_choice_0_or_1);
				else
					ImGui::Checkbox("RidgeRegression", &data->opt_choice_0_or_1);
			}
				
			
			if (ImGui::Button("Fit Data")&& data->points.size() >1)
			{					
				data->fitted_points.clear();
				if(data->opt_enable_insert)
					FitInputDataInsert(data->points, data->opt_choice_0_or_1,data->fitted_points);
				else
					FitInputDataApproach(data->points, pDim , data->opt_choice_0_or_1,data->fitted_points);
			}
			
			if (ImGui::Button("Clear Data"))
			{
				data->points.clear();
				data->fitted_points.clear();
			}

			ImGui::Text("Mouse Left: drag to add lines,\nMouse Right: drag to scroll, click for context menu.");
			
						
			ImVec2 canvas_p0 = ImGui::GetCursorScreenPos();      // ImDrawList API uses screen coordinates!
			ImVec2 canvas_sz = ImGui::GetContentRegionAvail();   // Resize canvas to what's available
			if (canvas_sz.x < 50.0f) canvas_sz.x = 50.0f;
			if (canvas_sz.y < 50.0f) canvas_sz.y = 50.0f;
			ImVec2 canvas_p1 = ImVec2(canvas_p0.x + canvas_sz.x, canvas_p0.y + canvas_sz.y);

			// Draw border and background color
			ImGuiIO& io = ImGui::GetIO();
			ImDrawList* draw_list = ImGui::GetWindowDrawList();
			draw_list->AddRectFilled(canvas_p0, canvas_p1, IM_COL32(50, 50, 50, 255));
			draw_list->AddRect(canvas_p0, canvas_p1, IM_COL32(255, 255, 255, 255));

			// This will catch our interactions
			ImGui::InvisibleButton("canvas", canvas_sz, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
			const bool is_hovered = ImGui::IsItemHovered(); // Hovered
			const bool is_active = ImGui::IsItemActive();   // Held
			const ImVec2 origin(canvas_p0.x + data->scrolling[0], canvas_p0.y + data->scrolling[1]); // Lock scrolled origin
			const pointf2 mouse_pos_in_canvas((io.MousePos.x - origin.x)/ dataScale, (io.MousePos.y - origin.y)/ dataScale);

			// Add first and second point
			if (is_hovered && !data->adding_line && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
			{
				data->points.push_back(mouse_pos_in_canvas);
			}
			if (data->adding_line)
			{
				data->points.back() = mouse_pos_in_canvas;
				if (!ImGui::IsMouseDown(ImGuiMouseButton_Left))
					data->adding_line = false;
			}
			// Pan (we use a zero mouse threshold when there's no context menu)
			// You may decide to make that threshold dynamic based on whether the mouse is hovering something etc.
			const float mouse_threshold_for_pan = data->opt_enable_context_menu ? -1.0f : 0.0f;
			if (is_active && ImGui::IsMouseDragging(ImGuiMouseButton_Right, mouse_threshold_for_pan))
			{
				data->scrolling[0] += io.MouseDelta.x;
				data->scrolling[1] += io.MouseDelta.y;
			}

			// Context menu (under default mouse threshold)
			ImVec2 drag_delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Right);
			if (data->opt_enable_context_menu && ImGui::IsMouseReleased(ImGuiMouseButton_Right) && drag_delta.x == 0.0f && drag_delta.y == 0.0f)
				ImGui::OpenPopupContextItem("context");
			if (ImGui::BeginPopup("context"))
			{
				if (data->adding_line)
					data->points.resize(data->points.size() - 2);
				data->adding_line = false;
				if (ImGui::MenuItem("Remove one", NULL, false, data->points.size() > 0)) { data->points.resize(data->points.size() - 2); }
				if (ImGui::MenuItem("Remove all", NULL, false, data->points.size() > 0)) { data->points.clear(); }
				ImGui::EndPopup();
			}

			// Draw grid + all lines in the canvas
			draw_list->PushClipRect(canvas_p0, canvas_p1, true);
			if (data->opt_enable_grid)
			{
				const float GRID_STEP = 64.0f;
				for (float x = fmodf(data->scrolling[0], GRID_STEP); x < canvas_sz.x; x += GRID_STEP)
					draw_list->AddLine(ImVec2(canvas_p0.x + x, canvas_p0.y), ImVec2(canvas_p0.x + x, canvas_p1.y), IM_COL32(200, 200, 200, 40));
				for (float y = fmodf(data->scrolling[1], GRID_STEP); y < canvas_sz.y; y += GRID_STEP)
					draw_list->AddLine(ImVec2(canvas_p0.x, canvas_p0.y + y), ImVec2(canvas_p1.x, canvas_p0.y + y), IM_COL32(200, 200, 200, 40));
			}
			//for (int n = 0; n < data->points.size(); n += 2)
			//	draw_list->AddLine(ImVec2(origin.x + data->points[n][0], origin.y + data->points[n][1]), ImVec2(origin.x + data->points[n + 1][0], origin.y + data->points[n + 1][1]), IM_COL32(255, 255, 0, 255), 2.0f);
			for (int n = 0; n < data->points.size(); ++n) {
				draw_list->AddCircleFilled(ImVec2(origin.x+data->points[n][0] * dataScale, origin.y + data->points[n][1] * dataScale), 3, IM_COL32(255, 0, 0, 255));
			}
			for (int n = 0; n < ((int)data->fitted_points.size())-1; ++n){
				draw_list->AddLine(ImVec2(origin.x + data->fitted_points[n][0]* dataScale, origin.y + data->fitted_points[n][1] * dataScale), ImVec2(origin.x + data->fitted_points[n + 1][0] * dataScale, origin.y + data->fitted_points[n + 1][1] * dataScale), IM_COL32(255, 255, 0, 255), 2.0f);
			}
				
			draw_list->PopClipRect();
		}

		ImGui::End();
	});
}
