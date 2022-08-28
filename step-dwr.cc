/**
  Thomas Wick
  Leibniz University Hannover
  Institute for Applied Mathematics
 
  Date:   Aug 15, 2022
  E-Mail: thomas.wick@ifam.uni-hannover.de
  Web:    www.thomaswick.org
 

  This code is based on the deal.II.9.1.0 version and
  licensed under the "GNU Lesser General Public License (LGPL)"
  with all information in LICENSE.
  Copyright 2022: Thomas Wick 
 
 
  Purpose of this program:
  1. Compute vector-Poisson on the unit square
  2. Estimate error with a posteriori goal-oriented error 
     with respect to some goal functional
  3. Adaptive mesh refinement using a parition-of-unity (PU) localization, i.e.,
     localization is achieved with adjoint sensitivity measures that are plugged into 
     the weak form
  4. Observe error reduction (in the goal functional) and Ieff in order to verify the program
 
 
  Main literature / Background
  1. R. Becker, R. Rannacher; An optimal control approach to a posteriori error 
     estimation in finite element methods, Acta Numerica, pp. 1-102, 2001
  2. W. Bangerth, R. Rannacher; Adaptive Finite Element Methods for Differential Equations,
     Lectures in Mathematics, ETH Z\"urich, Birkh\"auser, 2003
  3. T. Richter, T. Wick; Variational localizations of the dual weighted residual estimator,
     Journal of Computational and Applied Mathematics, Vol. 279, pp. 192-208, 2015


*/

//////////////////////////////////////////////////////////////////////
// Include files
//--------------

// The first step, as always, is to include
// the functionality of these 
// deal.II library files and some C++ header
// files.
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/timer.h>  

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>


#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_tools.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/solution_transfer.h>

//if DEAL_II_VERSION_GTE(9,0,0)
#include <deal.II/grid/manifold_lib.h>


// C++
#include <fstream>
#include <sstream>

// At the end of this top-matter, we import
// all deal.II names into the global
// namespace:				
using namespace dealii;


/////////////////////////////////////////////////////////////////////////////////////////
// Main file.
//
// In the next class, we define the main problem at hand.
// Here, we implement the top-level logic of solving a
// DWR (dual weighted residual) program.
//
// The  program is organized as follows. First, we set up
// runtime parameters and the system as done in other deal.II tutorial steps. 
// Then, we assemble
// the system matrix (Jacobian of Newton's method) 
// and system right hand side (residual of Newton's method) for the non-linear
// system. Two functions for the boundary values are provided because
// we are only supposed to apply boundary values in the first Newton step. In the
// subsequent Newton steps all Dirichlet values have to be equal zero.
// Afterwards, the routines for solving the linear 
// system and the Newton iteration are self-explaining. The following
// function is standard in deal.II tutorial steps:
// writing the solutions to graphical output. 
//
// The last functions provide the framework to compute 
// functional values of interest (goal functionals) to which we
// carry out local mesh refinement.
template <int dim>
class Poisson_PU_DWR_Problem 
{
public:
  
  Poisson_PU_DWR_Problem (const unsigned int degree);
  ~Poisson_PU_DWR_Problem (); 
  void run ();
  
private:

  // Setup of material parameters, 
  // spatial grid, etc. for primal and adjoint problems
  void set_runtime_parameters ();


  // Primal
  // Create system matrix, rhs and distribute degrees of freedom.
  void setup_system_primal ();

  // Assemble left and right hand side for Newton's method
  void assemble_matrix_primal ();   
  void assemble_rhs_primal ();

  // Boundary conditions (bc)
  void set_initial_bc_primal ();
  void set_newton_bc_primal ();

  // Linear solver
  void solve_primal ();

  // Nonlinear solver
  void newton_iteration_primal();


  //// Adjoint equations and their solution
  void setup_system_adjoint ();
  void assemble_matrix_adjoint ();   
  void assemble_rhs_adjoint_boundary ();
  void assemble_rhs_adjoint_point_value ();
  void set_bc_adjoint ();
  void solve_adjoint ();

  // Graphical visualization of output
  void output_results (const unsigned int refinement_cycle) const;


  // Evaluation of functional values  
  double compute_point_value (Point<dim> p,
			      const unsigned int component) const;
  
  void compute_boundary_tensor ();
  void compute_functional_values ();

  // Local adaptive mesh refinement
  void refine_mesh();
  double compute_error_indicators_a_la_PU_DWR (const unsigned int refinement_cycle);
  double refine_average_with_PU_DWR (const unsigned int refinement_cycle);


  // Global refinement
  const unsigned int   degree;
  
  Triangulation<dim>   triangulation;

  // Primal solution
  FESystem<dim>        fe_primal;
  DoFHandler<dim>      dof_handler_primal;

  ConstraintMatrix     constraints_primal;
  
  BlockSparsityPattern      sparsity_pattern_primal; 
  BlockSparseMatrix<double> system_matrix_primal; 
  
  BlockVector<double> solution_primal, newton_update_primal, old_timestep_solution_primal;
  BlockVector<double> system_rhs_primal;

  SparseDirectUMFPACK A_direct_primal;
 
  // Adjoint solution
  FESystem<dim>        fe_adjoint;
  DoFHandler<dim>      dof_handler_adjoint;

  ConstraintMatrix     constraints_adjoint;
  
  BlockSparsityPattern      sparsity_pattern_adjoint; 
  BlockSparseMatrix<double> system_matrix_adjoint; 
  
  BlockVector<double> solution_adjoint, old_timestep_solution_adjoint;
  BlockVector<double> system_rhs_adjoint;

  // PU for PU-DWR localization
  FESystem<dim>        fe_pou;
  DoFHandler<dim>      dof_handler_pou;
  Vector<float>        error_indicators;


  // Measuring CPU times
  TimerOutput         timer;
  
  // Right hand side forces and values
  double force_x, force_y;
  
  std::string adjoint_rhs;
  std::ofstream file_gnuplot;
  std::ofstream file;

  // DWR refinement stuff
  unsigned int max_no_refinement_cycles, max_no_degrees_of_freedom; 
  double TOL_DWR_estimator, lower_bound_newton_residuum; 

  unsigned int refinement_strategy;
  
  //Reference values
  double reference_value, reference_value_u_point, exact_error_local;
  

};


// The constructor of this class is comparable 
// to other tutorials steps, e.g., step-3, step-22, and step-31. 
// We are going to use Q1 elements for the primal variable (vector-valued displacements)
// and Q2 elements for the adjoint variable.
template <int dim>
Poisson_PU_DWR_Problem<dim>::Poisson_PU_DWR_Problem (const unsigned int degree)
                :
                degree (degree),
		triangulation (Triangulation<dim>::maximum_smoothing),
		//triangulation (Triangulation<dim>::none),

                fe_primal (FE_Q<dim>(1), dim),  // primal variable; displacements                  
                dof_handler_primal (triangulation),

		// Info: adjoint degree must be higher than primal
		// degree as usual for DWR-based error estimation (see step-14)
		fe_adjoint (FE_Q<dim>(2), dim),  // adjoint variable                 
                dof_handler_adjoint (triangulation),

                // Lowest order FE to gather neighboring information
		// see Richter/Wick; JCAM, 2015
		fe_pou (FE_Q<dim>(1), 1),
		dof_handler_pou (triangulation),

		timer (std::cout, TimerOutput::summary, TimerOutput::cpu_times)		
{}


// This is the standard destructor.
template <int dim>
Poisson_PU_DWR_Problem<dim>::~Poisson_PU_DWR_Problem () 
{}


// In this method, we set up runtime parameters that 
// could also come from a parameter file. 
// The user is invited to change these values to obtain
// other results. 
template <int dim>
void Poisson_PU_DWR_Problem<dim>::set_runtime_parameters ()
{

  // As goal functionals, we have implemented 
  // a boundary line integration and a point value evaluation
  adjoint_rhs = "point"; // "point" or "boundary" as choices
  max_no_refinement_cycles = 6;
  max_no_degrees_of_freedom = 2.0e+6;
  TOL_DWR_estimator = 1.0e-10;

  // Setting tolerance for primal Newton solver
  // Info: despite PDE (Poisson) is linear, we use 
  //       a nonlinear solver to allow for 
  //       an easy extension to nonlinear problems
  lower_bound_newton_residuum = 1.0e-8;

  // 0: global refinement
  // 1: PU DWR
  // 2: as in 0: global, but can be easily modified to something else
  refinement_strategy = 1;

  // Right hand side of the PDE
  // Info: we implemented vector-Laplace (Poisson) in order to 
  //       allow for an easy extension to PDE systems.
  force_x = -1.0;
  force_y = 0.0; 

  // Mesh generation
  GridGenerator::hyper_cube (triangulation, 0, 1);
  triangulation.refine_global (1); 

  // For DWR output (effectivity indices, error behavior, etc.) into files
  std::string filename = "dwr_results.txt";
  std::string filename_gnuplot = "dwr_results_gp.txt";
  file.open(filename.c_str());
  file_gnuplot.open(filename_gnuplot.c_str());

  file << "Dofs" << "\t" << "Exact err" << "\t" << "Est err   " << "\t" << "Est ind   " << "\t" << "Eff" << "\t" << "Ind" << "\n";
  file.flush();
 
}

/////////////////////////////////////////////////////////////////////////////////////////
// Primal problem implementation.


// This function is similar to many deal.II tuturial steps.
template <int dim>
void Poisson_PU_DWR_Problem<dim>::setup_system_primal ()
{
  timer.enter_section("Setup system.");

  system_matrix_primal.clear ();
  
  dof_handler_primal.distribute_dofs (fe_primal);  
  DoFRenumbering::Cuthill_McKee (dof_handler_primal);

  // We are dealing with 2 components since we implement a vector-Laplace
  std::vector<unsigned int> block_component (2,0);
 
  DoFRenumbering::component_wise (dof_handler_primal, block_component);

  {				 
    constraints_primal.clear ();
    set_newton_bc_primal ();
    DoFTools::make_hanging_node_constraints (dof_handler_primal,
					     constraints_primal);
  }
  constraints_primal.close ();
  
  // Two blocks: velocity and pressure
  std::vector<unsigned int> dofs_per_block (1);
  DoFTools::count_dofs_per_block (dof_handler_primal, dofs_per_block, block_component);  
  const unsigned int n_u = dofs_per_block[0];

  std::cout << "Elements:\t"
            << triangulation.n_active_cells()
            << std::endl  	  
            << "DoFs (primal):\t"
            << dof_handler_primal.n_dofs()
            << " (" << n_u <<  ')'
            << std::endl;


 
      
 {
    BlockDynamicSparsityPattern csp (1,1);

    csp.block(0,0).reinit (n_u, n_u);
 
    csp.collect_sizes();    

    DoFTools::make_sparsity_pattern (dof_handler_primal, csp, constraints_primal, false);

    sparsity_pattern_primal.copy_from (csp);
  }
 
 system_matrix_primal.reinit (sparsity_pattern_primal);

  // Actual solution 
  solution_primal.reinit (1);
  solution_primal.block(0).reinit (n_u);
 
  solution_primal.collect_sizes ();
 
  // Updates for Newton's method (since we solve generically with a Newton-type method)
  newton_update_primal.reinit (1);
  newton_update_primal.block(0).reinit (n_u);
 
  newton_update_primal.collect_sizes ();
 
  // Residual for  Newton's method
  system_rhs_primal.reinit (1);
  system_rhs_primal.block(0).reinit (n_u);

  system_rhs_primal.collect_sizes ();

  timer.exit_section(); 
}


// In this function, we assemble the Jacobian matrix
// for the Newton iteration. 
//
// Assembling of the inner most loop is treated with help of 
// the fe.system_to_component_index(j).first function from
// the library. 
// Using this function makes the assembling process much faster
// than running over all local degrees of freedom. 
template <int dim>
void Poisson_PU_DWR_Problem<dim>::assemble_matrix_primal ()
{
  timer.enter_section("Assemble primal matrix.");
  system_matrix_primal = 0;
     
  QGauss<dim>   quadrature_formula(degree+2);  
  QGauss<dim-1> face_quadrature_formula(degree+2);

  FEValues<dim> fe_values (fe_primal, quadrature_formula,
                           update_values    |
                           update_quadrature_points  |
                           update_JxW_values |
                           update_gradients);
  
  FEFaceValues<dim> fe_face_values (fe_primal, face_quadrature_formula, 
				    update_values         | update_quadrature_points  |
				    update_normal_vectors | update_gradients |
				    update_JxW_values);
   
  const unsigned int   dofs_per_cell   = fe_primal.dofs_per_cell;
  
  const unsigned int   n_q_points      = quadrature_formula.size();
  const unsigned int n_face_q_points   = face_quadrature_formula.size();

  FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);

  std::vector<unsigned int> local_dof_indices (dofs_per_cell); 
		

  // Now, we are going to use the 
  // FEValuesExtractors to determine
  // the four principle variables
  const FEValuesExtractors::Vector displacements (0); 

  // We declare Vectors and Tensors for 
  // the solutions at the previous Newton iteration:
  std::vector<Vector<double> > old_solution_values (n_q_points, 
				 		    Vector<double>(dim));

  std::vector<std::vector<Tensor<1,dim> > > old_solution_grads (n_q_points, 
								std::vector<Tensor<1,dim> > (dim));

  std::vector<Vector<double> >  old_solution_face_values (n_face_q_points, 
							  Vector<double>(dim));
       
  std::vector<std::vector<Tensor<1,dim> > > old_solution_face_grads (n_face_q_points, 
								     std::vector<Tensor<1,dim> > (dim));
    
   
  // Declaring test functions:
  std::vector<Tensor<1,dim> > phi_i_u (dofs_per_cell); 
  std::vector<Tensor<2,dim> > phi_i_grads_u(dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler_primal.begin_active(),
    endc = dof_handler_primal.end();
  
  // Looping over all elements
  for (; cell!=endc; ++cell)
    { 
      fe_values.reinit (cell);
      local_matrix = 0;
      
      // Previous Newton iteration values
      fe_values.get_function_values (solution_primal, old_solution_values);
      fe_values.get_function_gradients (solution_primal, old_solution_grads);

      for (unsigned int q=0; q<n_q_points; ++q)
	{
	  for (unsigned int k=0; k<dofs_per_cell; ++k)
	    {
	      phi_i_u[k]       = fe_values[displacements].value (k, q);
	      phi_i_grads_u[k] = fe_values[displacements].gradient (k, q);
	    }
	  
	      // We build values, vectors, and tensors
	      // from information of the previous Newton step.
	      Tensor<1,dim> u; 
	      u[0] = old_solution_values[q](0);
	      u[1] = old_solution_values[q](1);

	      Tensor<2,dim> grad_u;
	      grad_u[0][0] = old_solution_grads[q][0][0];
	      grad_u[0][1] = old_solution_grads[q][0][1];
	      grad_u[1][0] = old_solution_grads[q][1][0];
	      grad_u[1][1] = old_solution_grads[q][1][1];
	      
	      
	      // Outer loop for dofs
	      for (unsigned int i=0; i<dofs_per_cell; ++i)
		{
		  Tensor<2,dim> grad_u_LinU;
		  grad_u_LinU[0][0] = phi_i_grads_u[i][0][0];
		  grad_u_LinU[0][1] = phi_i_grads_u[i][0][1];
		  grad_u_LinU[1][0] = phi_i_grads_u[i][1][0];
		  grad_u_LinU[1][1] = phi_i_grads_u[i][1][1];
						      
		  // Inner loop for dofs
		  for (unsigned int j=0; j<dofs_per_cell; ++j)
		    {	
		      // Assemble PDE
		      const unsigned int comp_j = fe_primal.system_to_component_index(j).first; 
		      if (comp_j == 0 || comp_j == 1)
			{		
			  local_matrix(j,i) += (scalar_product(grad_u_LinU, phi_i_grads_u[j])
						) * fe_values.JxW(q);
			} // end comp					    		    
		    } // end j dofs	     
		} // end i dofs
	    } // end n_q_points 
	  
	  // Write local entries into global matrix
	  cell->get_dof_indices (local_dof_indices);
	  constraints_primal.distribute_local_to_global (local_matrix, local_dof_indices,
						  system_matrix_primal);

      
    }   // end element
  
  timer.exit_section();
}



// In this function we assemble the semi-linear 
// of the right hand side of Newton's method (its residual).
// The framework is in principal the same as for the 
// system matrix.
template <int dim>
void
Poisson_PU_DWR_Problem<dim>::assemble_rhs_primal ()
{
  timer.enter_section("Assemble primal rhs.");
  system_rhs_primal = 0;
  
  QGauss<dim>   quadrature_formula(degree+2);
  QGauss<dim-1> face_quadrature_formula(degree+2);

  FEValues<dim> fe_values (fe_primal, quadrature_formula,
                           update_values    |
                           update_quadrature_points  |
                           update_JxW_values |
                           update_gradients);

  FEFaceValues<dim> fe_face_values (fe_primal, face_quadrature_formula, 
				    update_values         | update_quadrature_points  |
				    update_normal_vectors | update_gradients |
				    update_JxW_values);

  const unsigned int   dofs_per_cell   = fe_primal.dofs_per_cell;
  
  const unsigned int   n_q_points      = quadrature_formula.size();
  const unsigned int n_face_q_points   = face_quadrature_formula.size();
 
  Vector<double>       local_rhs (dofs_per_cell);

  std::vector<unsigned int> local_dof_indices (dofs_per_cell);
  
  const FEValuesExtractors::Vector displacements (0);
 
  std::vector<Vector<double> > 
    old_solution_values (n_q_points, Vector<double>(dim));

  std::vector<std::vector<Tensor<1,dim> > > 
    old_solution_grads (n_q_points, std::vector<Tensor<1,dim> > (dim));


  std::vector<Vector<double> > 
    old_solution_face_values (n_face_q_points, Vector<double>(dim));
  
  std::vector<std::vector<Tensor<1,dim> > > 
    old_solution_face_grads (n_face_q_points, std::vector<Tensor<1,dim> > (dim));
  
   
  
  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler_primal.begin_active(),
    endc = dof_handler_primal.end();

  for (; cell!=endc; ++cell)
    { 
      fe_values.reinit (cell);	 
      local_rhs = 0;   	
      
      // Previous Newton iteration
      fe_values.get_function_values (solution_primal, old_solution_values);
      fe_values.get_function_gradients (solution_primal, old_solution_grads);
            
      for (unsigned int q=0; q<n_q_points; ++q)
	{
	  // Right hand side 
	  Tensor<1,dim> force;
	  force.clear();
	  force[0] = force_x;
	  force[1] = force_y;	  
	  
	  Tensor<1,dim> u; 
	  u[0] = old_solution_values[q](0);
	  u[1] = old_solution_values[q](1);
	  
	  Tensor<2,dim> grad_u;
	  grad_u[0][0] = old_solution_grads[q][0][0];
	  grad_u[0][1] = old_solution_grads[q][0][1];
	  grad_u[1][0] = old_solution_grads[q][1][0];
	  grad_u[1][1] = old_solution_grads[q][1][1];

	  for (unsigned int i=0; i<dofs_per_cell; ++i)
	    {
	      const unsigned int comp_i = fe_primal.system_to_component_index(i).first; 
	      if (comp_i == 0 || comp_i == 1)
		{   		  
		  const Tensor<1,dim> phi_i_u = fe_values[displacements].value (i, q);
		  const Tensor<2,dim> phi_i_grads_u = fe_values[displacements].gradient (i, q);
		  
		  local_rhs(i) -= (scalar_product(grad_u, phi_i_grads_u)
				   - force * phi_i_u
				   ) *  fe_values.JxW(q);
		  
		} // end comp		
	    } // end i dofs   	     	   
	} // close n_q_points  
	  
	  
      cell->get_dof_indices (local_dof_indices);
      constraints_primal.distribute_local_to_global (local_rhs, local_dof_indices,
						     system_rhs_primal);
      
    }  // end element
      
  timer.exit_section();
}


// Here, we impose boundary conditions for the whole system. 
// Since we use a nonlinear solver (Newton), we later need 
// a second function that sets for each Newton update non-homogeneous Dirichlet
// conditions to homogeneous ones.
// Info: in the current implementation, we only have homogeneous 
//       Dirichlet conditions, which however could be easily changed.
template <int dim>
void
Poisson_PU_DWR_Problem<dim>::set_initial_bc_primal ()
{ 
    std::map<unsigned int,double> boundary_values;  
    std::vector<bool> component_mask (dim, true);
 
    VectorTools::interpolate_boundary_values (dof_handler_primal,
					      0,
					      ZeroFunction<dim>(dim),  
					      boundary_values,
					      component_mask);    


    VectorTools::interpolate_boundary_values (dof_handler_primal,
                                              2,
					      ZeroFunction<dim>(dim),  
                                              boundary_values,
                                              component_mask);


    VectorTools::interpolate_boundary_values (dof_handler_primal,
                                              3,
					      ZeroFunction<dim>(dim),  
                                              boundary_values,
                                              component_mask);
    
    component_mask[0] = true;
    component_mask[1] = true;   
    
    VectorTools::interpolate_boundary_values (dof_handler_primal,
					      1,
					      ZeroFunction<dim>(dim),  
					      boundary_values,
					      component_mask);
    
    for (typename std::map<unsigned int, double>::const_iterator
	   i = boundary_values.begin();
	 i != boundary_values.end();
	 ++i)
      solution_primal(i->first) = i->second;
    
}

// This function applies boundary conditions 
// to the Newton iteration steps. For all variables that
// have Dirichlet conditions on some (or all) parts
// of the outer boundary, we apply zero-Dirichlet
// conditions, now. 
template <int dim>
void
Poisson_PU_DWR_Problem<dim>::set_newton_bc_primal ()
{
    std::vector<bool> component_mask (dim, true);
   
    VectorTools::interpolate_boundary_values (dof_handler_primal,
					      0,
					      ZeroFunction<dim>(dim),                           
					      constraints_primal,
					      component_mask); 

    VectorTools::interpolate_boundary_values (dof_handler_primal,
                                              2,
					      ZeroFunction<dim>(dim),  
                                              constraints_primal,
                                              component_mask);
    
    VectorTools::interpolate_boundary_values (dof_handler_primal,
                                              3,
					      ZeroFunction<dim>(dim),  
                                              constraints_primal,
                                              component_mask);

    component_mask[0] = true;
    component_mask[1] = true;
    
    VectorTools::interpolate_boundary_values (dof_handler_primal,
					      1,
					      ZeroFunction<dim>(dim),  
					      constraints_primal,
					      component_mask);
}  

// In this function, we solve the linear systems
// inside the nonlinear Newton iteration. For simplicity we
// use a direct solver from UMFPACK. For Poisson (Laplace) simply 
// a CG method (step-3) or multigrid (step-16) could be used as well.
template <int dim>
void 
Poisson_PU_DWR_Problem<dim>::solve_primal () 
{
  timer.enter_section("Solve primal linear system.");
  Vector<double> sol, rhs;    
  sol = newton_update_primal;    
  rhs = system_rhs_primal;
    
  A_direct_primal.vmult(sol,rhs); 
  newton_update_primal = sol;
  
  constraints_primal.distribute (newton_update_primal);
  timer.exit_section();
}

// This is the Newton iteration with simple linesearch backtracking 
// to solve the 
// non-linear system of equations. First, we declare some
// standard parameters of the solution method. Addionally,
// we also implement an easy line search algorithm. 
template <int dim>
void Poisson_PU_DWR_Problem<dim>::newton_iteration_primal () 
					       
{ 
  Timer timer_newton;
  Timer timer_newton_global;
  timer_newton_global.start();

  const unsigned int max_no_newton_steps  = 10;

  // Decision whether the system matrix should be build
  // at each Newton step
  const double nonlinear_rho = 0.1; 
 
  // Line search parameters
  unsigned int line_search_step;
  const unsigned int  max_no_line_search_steps = 5;
  const double line_search_damping = 0.6;
  double new_newton_residuum;
  
  // Application of the initial boundary conditions to the 
  // variational equations:
  set_initial_bc_primal ();
  assemble_rhs_primal();

  double newton_residuum = system_rhs_primal.linfty_norm(); 
  double old_newton_residuum= newton_residuum;
  unsigned int newton_step = 1;
   
  if (newton_residuum < lower_bound_newton_residuum)
    {
      std::cout << '\t' 
		<< std::scientific 
		<< newton_residuum 
		<< std::endl;     
    }
  
  while (newton_residuum > lower_bound_newton_residuum &&
	 newton_step < max_no_newton_steps)
    {
      timer_newton.start();
      old_newton_residuum = newton_residuum;
      
      assemble_rhs_primal();
      newton_residuum = system_rhs_primal.linfty_norm();

      if (newton_residuum < lower_bound_newton_residuum)
	{
	  std::cout << '\t' 
		    << std::scientific 
		    << newton_residuum << std::endl;
	  break;
	}
  
      if (newton_residuum/old_newton_residuum > nonlinear_rho)
	{
	  assemble_matrix_primal ();	
	  A_direct_primal.factorize(system_matrix_primal);  
	}

      // Solve Ax = b
      solve_primal ();	  
        
      line_search_step = 0;	  
      for ( ; 
	    line_search_step < max_no_line_search_steps; 
	    ++line_search_step)
	{	     					 
	  solution_primal += newton_update_primal;
	  
	  assemble_rhs_primal ();			
	  new_newton_residuum = system_rhs_primal.linfty_norm();
	  
	  if (new_newton_residuum < newton_residuum)
	      break;
	  else 	  
	    solution_primal -= newton_update_primal;
	  
	  newton_update_primal *= line_search_damping;
	}	   
     
      timer_newton.stop();
      
      std::cout << std::setprecision(5) <<newton_step << '\t' 
		<< std::scientific << newton_residuum << '\t'
		<< std::scientific << newton_residuum/old_newton_residuum  <<'\t' ;
      if (newton_residuum/old_newton_residuum > nonlinear_rho)
	std::cout << "r" << '\t' ;
      else 
	std::cout << " " << '\t' ;
      std::cout << line_search_step  << '\t' 
		<< std::scientific << timer_newton ()
		<< std::endl;


      // Updates
      timer_newton.reset();
      newton_step++;      
    }

  timer_newton_global.stop();
  std::cout << "CPU time solving primal system:  " << timer_newton_global() << std::endl;
  timer_newton_global.reset();


}



/////////////////////////////////////////////////////////////////////////////////////////
// Adjoint problem implementation.


// This function is similar to many deal.II tuturial steps.
template <int dim>
void Poisson_PU_DWR_Problem<dim>::setup_system_adjoint ()
{
  timer.enter_section("Setup adjoint system.");

  system_matrix_adjoint.clear ();
  
  dof_handler_adjoint.distribute_dofs (fe_adjoint);  
  DoFRenumbering::Cuthill_McKee (dof_handler_adjoint);

  // Declaring components           1
  std::vector<unsigned int> block_component (2,0);
 
  DoFRenumbering::component_wise (dof_handler_adjoint, block_component);

  {				 
    constraints_adjoint.clear ();
     DoFTools::make_hanging_node_constraints (dof_handler_adjoint,
					     constraints_adjoint);
    set_bc_adjoint ();
  }
  constraints_adjoint.close ();
  
  std::vector<unsigned int> dofs_per_block (1);
  DoFTools::count_dofs_per_block (dof_handler_adjoint, dofs_per_block, block_component);  
  const unsigned int n_u = dofs_per_block[0];

  std::cout << "DoFs (adjoint):\t"
            << dof_handler_adjoint.n_dofs()
            << " (" << n_u <<  ')'
            << std::endl;


 
      
 {
   BlockDynamicSparsityPattern csp (1,1);

   csp.block(0,0).reinit (n_u, n_u);
   
   csp.collect_sizes();    
   
   DoFTools::make_sparsity_pattern (dof_handler_adjoint, csp, constraints_adjoint, false);
   
   sparsity_pattern_adjoint.copy_from (csp);
 }
 
 system_matrix_adjoint.reinit (sparsity_pattern_adjoint);

  // Current solution 
  solution_adjoint.reinit (1);
  solution_adjoint.block(0).reinit (n_u);
 
  solution_adjoint.collect_sizes ();
 
  // Residual for  Newton's method
  system_rhs_adjoint.reinit (1);
  system_rhs_adjoint.block(0).reinit (n_u);

  system_rhs_adjoint.collect_sizes ();

  timer.exit_section(); 
}




template <int dim>
void Poisson_PU_DWR_Problem<dim>::assemble_matrix_adjoint ()
{
  timer.enter_section("Assemble adjoint matrix.");
  system_matrix_adjoint = 0;
     
  // Choose quadrature rule sufficiently high with respect 
  // to the finite element choice.
  QGauss<dim>   quadrature_formula(degree+4);  
  QGauss<dim-1> face_quadrature_formula(degree+4);

  FEValues<dim> fe_values (fe_adjoint, quadrature_formula,
                           update_values    |
                           update_quadrature_points  |
                           update_JxW_values |
                           update_gradients);
  
  FEFaceValues<dim> fe_face_values (fe_adjoint, face_quadrature_formula, 
				    update_values         | update_quadrature_points  |
				    update_normal_vectors | update_gradients |
				    update_JxW_values);


 FEValues<dim> fe_values_primal (fe_primal, quadrature_formula,
                           update_values    |
                           update_quadrature_points  |
                           update_JxW_values |
                           update_gradients);

   
  const unsigned int   dofs_per_cell   = fe_adjoint.dofs_per_cell;
  
  const unsigned int   n_q_points      = quadrature_formula.size();
  const unsigned int n_face_q_points   = face_quadrature_formula.size();

  FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);

  std::vector<unsigned int> local_dof_indices (dofs_per_cell); 
		

  // Now, we are going to use the 
  // FEValuesExtractors to determine
  // the principle variables
  const FEValuesExtractors::Vector displacements (0);

  // We declare Vectors and Tensors for 
  // the solutions at the previous Newton iteration:
  std::vector<Vector<double> > old_solution_values (n_q_points, 
				 		    Vector<double>(dim));

  std::vector<std::vector<Tensor<1,dim> > > old_solution_grads (n_q_points, 
								std::vector<Tensor<1,dim> > (dim));

  std::vector<Vector<double> >  old_solution_face_values (n_face_q_points, 
							  Vector<double>(dim));
       
  std::vector<std::vector<Tensor<1,dim> > > old_solution_face_grads (n_face_q_points, 
								     std::vector<Tensor<1,dim> > (dim));



  // Primal solution values in case we are dealing with nonlinear
  // problems, where the primal solution must be available as coefficient 
  // function for the adjoint.
 std::vector<Vector<double> > old_solution_values_primal (n_q_points, 
							  Vector<double>(dim));

 std::vector<std::vector<Tensor<1,dim> > > old_solution_grads_primal (n_q_points, 
								      std::vector<Tensor<1,dim> > (dim));

    
   
  // Declaring test functions:
  std::vector<Tensor<1,dim> > phi_i_u (dofs_per_cell); 
  std::vector<Tensor<2,dim> > phi_i_grads_u(dofs_per_cell);
 				     				   
  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler_adjoint.begin_active(),
    endc = dof_handler_adjoint.end();

  typename DoFHandler<dim>::active_cell_iterator
    cell_primal = dof_handler_primal.begin_active();
  
  for (; cell!=endc; ++cell, ++cell_primal)
    { 
      fe_values.reinit (cell);
      fe_values_primal.reinit (cell_primal);

      local_matrix = 0;
      
      
      // Previous Newton iteration values
      fe_values.get_function_values (solution_adjoint, old_solution_values);
      fe_values.get_function_gradients (solution_adjoint, old_solution_grads);

      // Solution values from the primal problem
      fe_values_primal.get_function_values (solution_primal, old_solution_values_primal);
      fe_values_primal.get_function_gradients (solution_primal, old_solution_grads_primal);
      
      
      // Next, we run over all elements
	  for (unsigned int q=0; q<n_q_points; ++q)
	    {
	      for (unsigned int k=0; k<dofs_per_cell; ++k)
		{
		  phi_i_u[k]       = fe_values[displacements].value (k, q);
		  phi_i_grads_u[k] = fe_values[displacements].gradient (k, q);
		}
	      
	      // Primal solution (only needed for nonlinear problems)
	      Tensor<1,dim> u_primal;
	      u_primal[0] = old_solution_values_primal[q](0);
	      u_primal[1] = old_solution_values_primal[q](1);

	      Tensor<2,dim> grad_u_primal;
	      grad_u_primal[0][0] = old_solution_grads_primal[q][0][0];
	      grad_u_primal[0][1] = old_solution_grads_primal[q][0][1];
	      grad_u_primal[1][0] = old_solution_grads_primal[q][1][0];
	      grad_u_primal[1][1] = old_solution_grads_primal[q][1][1];
	      
	      // Adjoint - really ever needed?
	      //Tensor<2,dim> grad_u;
	      //grad_u[0][0] = old_solution_grads[q][0][0];
	      //grad_u[0][1] = old_solution_grads[q][0][1];
	      //grad_u[1][0] = old_solution_grads[q][1][0];
	      //grad_u[1][1] = old_solution_grads[q][1][1];

	      // Outer loop for dofs
	      for (unsigned int i=0; i<dofs_per_cell; ++i)
		{
		  // Adjoint
		  Tensor<2,dim> grad_zu;
		  grad_zu[0][0] = phi_i_grads_u[i][0][0];
		  grad_zu[0][1] = phi_i_grads_u[i][0][1];
		  grad_zu[1][0] = phi_i_grads_u[i][1][0];
		  grad_zu[1][1] = phi_i_grads_u[i][1][1];

	
		  // Inner loop for dofs
		  for (unsigned int j=0; j<dofs_per_cell; ++j)
		    {	
		      // Info: In the adjoint matrix, the entries are flipped, i.e., 
		      // (i,j) (and not (j,i)) because the adjoint
		      // matrix is transposed by definition.

		      const unsigned int comp_j = fe_adjoint.system_to_component_index(j).first; 
		      if (comp_j == 0 || comp_j == 1)
			{		
			  local_matrix(i,j) += (scalar_product(grad_zu, phi_i_grads_u[j]) 	
						) * fe_values.JxW(q);
			} // end comp					    		   
		    }  // end j dofs  
		}  // end i dofs 
	    } // end n_q_points    
	  
	  // This is the same as for the primal problem
	  cell->get_dof_indices (local_dof_indices);
	  constraints_adjoint.distribute_local_to_global (local_matrix, local_dof_indices,
						  system_matrix_adjoint);

      
    } // end element  
  
  timer.exit_section();
}



// Implement goal functional right hand side: integral over some boundary
template <int dim>
void
Poisson_PU_DWR_Problem<dim>::assemble_rhs_adjoint_boundary ()
{
  timer.enter_section("Assemble adjoint rhs.");
  system_rhs_adjoint = 0;
  
  // Info: Quadrature degree must be sufficiently high
  QGauss<dim>   quadrature_formula(degree+4);
  QGauss<dim-1> face_quadrature_formula(degree+4);

  FEValues<dim> fe_values (fe_adjoint, quadrature_formula,
                           update_values    |
                           update_quadrature_points  |
                           update_JxW_values |
                           update_gradients);


  FEFaceValues<dim> fe_face_values (fe_adjoint, face_quadrature_formula, 
				    update_values         | update_quadrature_points  |
				    update_normal_vectors | update_gradients |
				    update_JxW_values);

  const unsigned int   dofs_per_cell   = fe_adjoint.dofs_per_cell;
  
  //const unsigned int   n_q_points      = quadrature_formula.size();
  const unsigned int n_face_q_points   = face_quadrature_formula.size();
 
  Vector<double>       local_rhs (dofs_per_cell);

  std::vector<unsigned int> local_dof_indices (dofs_per_cell);
  
  const FEValuesExtractors::Vector displacements (0);
  
  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler_adjoint.begin_active(),
    endc = dof_handler_adjoint.end();


  Tensor<1, 2> rhs_value;
  for (; cell!=endc; ++cell)
    { 
      fe_values.reinit (cell);	 
      local_rhs = 0;   	
          
	  // Assemble function on boundary #0
	  for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
	    {
	      if (cell->face(face)->at_boundary() && 		  
		  (cell->face(face)->boundary_id() == 0) 
		  )
		{
		  
		  fe_face_values.reinit (cell, face);
		  
		  for (unsigned int q=0; q<n_face_q_points; ++q)
		    {	
		      		     
		      for (unsigned int i=0; i<dofs_per_cell; ++i)
			{
			  const Tensor<2,dim> phi_i_grads_u = fe_face_values[displacements].gradient (i, q);

			  rhs_value -= phi_i_grads_u
			    * fe_face_values.normal_vector(q);
			  
			  // extract x-component (could be y-comp as well or the sum) and write
			  // this one into the rhs of the dual functional
			  local_rhs(i) += rhs_value[0] *
			    fe_face_values.JxW(q);
			    
			  
			}  // end i
		    }   // end face_n_q_points                                    
		} // end boundary id 
	    }  // end face terms

	  
	  cell->get_dof_indices (local_dof_indices);
	  constraints_adjoint.distribute_local_to_global (local_rhs, local_dof_indices,
						  system_rhs_adjoint);
	 
      
    }  // end cell
      
  timer.exit_section();
}


template <int dim>
void
Poisson_PU_DWR_Problem<dim>::assemble_rhs_adjoint_point_value ()
{
  // Not yet implemented.
  //abort();

  timer.enter_section("Assemble adjoint rhs.");
  system_rhs_adjoint = 0;
  
  Point<dim> evaluation_point(0.5,0.5);
  //Point<dim> evaluation_point_2(0.25,0.2);
  
  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler_adjoint.begin_active(),
    endc = dof_handler_adjoint.end();

  for (; cell!=endc; ++cell)
    for (unsigned int vertex=0;
	 vertex<GeometryInfo<dim>::vertices_per_cell;
	 ++vertex)
      {
	if (cell->vertex(vertex).distance(evaluation_point)
	    < cell->diameter()*1e-8)
	  {
	    // Find the degree of freedom that corresponds to
	    // this point in the mesh
	    // The first argument is are the vertex coordinates
	    // The second argument is the FE component: 0 (ux), 1(uy)
	    system_rhs_adjoint(cell->vertex_dof_index(vertex,0)) = 1;
	  }
      }
  

      
  timer.exit_section();



}




// Boundary conditions for the linear adjoint problem.
// All non-homogeneous Dirichlet conditions of 
// the primal problem are now zer-conditions in the 
// adjoint problem. 
// All primal Neumann conditions remain adjoint Neumann conditions.
// See optimization books for example.
template <int dim>
void
Poisson_PU_DWR_Problem<dim>::set_bc_adjoint ()
{
    std::vector<bool> component_mask (dim, true);

   
    VectorTools::interpolate_boundary_values (dof_handler_adjoint,
					      0,
					      ZeroFunction<dim>(dim),                                  
					      constraints_adjoint,
					      component_mask); 

    VectorTools::interpolate_boundary_values (dof_handler_adjoint,
                                              2,
					      ZeroFunction<dim>(dim),  
                                              constraints_adjoint,
                                              component_mask);
    
    VectorTools::interpolate_boundary_values (dof_handler_adjoint,
                                              3,
					      ZeroFunction<dim>(dim),  
                                              constraints_adjoint,
                                              component_mask);

    component_mask[0] = true;
    component_mask[1] = true;
    
    VectorTools::interpolate_boundary_values (dof_handler_adjoint,
					      1,
					      ZeroFunction<dim>(dim),  
					      constraints_adjoint,
					      component_mask);
}  



// In this function, we solve the linear adjoint system. 
// For simplicity we use a direct solver from UMFPACK.
template <int dim>
void 
Poisson_PU_DWR_Problem<dim>::solve_adjoint () 
{
  // Assembling linear adjoint system
  // Matrix is the derivative of the PDE 
  //  (i.e., the Jacobian of the primal problem; can be taken
  //  more or less as copy and paste from the assemble_matrix_primal function 
  assemble_matrix_adjoint ();  

  // The rhs depends on the specific goal functional 
  if (adjoint_rhs == "boundary")
    assemble_rhs_adjoint_boundary ();
  else if (adjoint_rhs == "point")
    assemble_rhs_adjoint_point_value (); 
  else
    {
      std::cout << "Aborting: Adjoint rhs not implemented." << std::endl;
      abort ();
    }
 
  // Solving the linear adjoint system
  Timer timer_solve_adjoint;
  timer_solve_adjoint.start();

  // Linear solution
  timer.enter_section("Solve linear adjoint system.");
  Vector<double> sol, rhs;    
  sol = solution_adjoint;    
  rhs = system_rhs_adjoint;
  
  SparseDirectUMFPACK A_direct;
  A_direct.factorize(system_matrix_adjoint);  
  
  A_direct.vmult(sol,rhs); 
  solution_adjoint = sol;
  
  constraints_adjoint.distribute (solution_adjoint);
  timer_solve_adjoint.stop();

  std::cout << "CPU time solving adjoint system: " << timer_solve_adjoint() << std::endl;

  timer_solve_adjoint.reset();

  timer.exit_section();



}



////////////////////////////////////////////////////////////////////////////////////////////
// 1. Output into vtk
// 2. Evaluation of goal functionals (quantities of interest)




// This function is known from almost all other 
// tutorial steps in deal.II. This we have different 
// finite elements working on the same triangulation, we first
// need to create a joint FE such that we can output all quantities 
// together.
template <int dim>
void
Poisson_PU_DWR_Problem<dim>::output_results (const unsigned int refinement_cycle)  const
{

  const FESystem<dim> joint_fe (fe_primal, 1,
				fe_adjoint, 1);
  DoFHandler<dim> joint_dof_handler (triangulation);
  joint_dof_handler.distribute_dofs (joint_fe);
  Assert (joint_dof_handler.n_dofs() ==
	  dof_handler_primal.n_dofs() + dof_handler_adjoint.n_dofs(),
	  ExcInternalError());

  Vector<double> joint_solution (joint_dof_handler.n_dofs());


 {
    std::vector<unsigned int> local_joint_dof_indices (joint_fe.dofs_per_cell);
    std::vector<unsigned int> local_dof_indices_primal (fe_primal.dofs_per_cell);
    std::vector<unsigned int> local_dof_indices_adjoint (fe_adjoint.dofs_per_cell);

      typename DoFHandler<dim>::active_cell_iterator
      joint_cell       = joint_dof_handler.begin_active(),
      joint_endc       = joint_dof_handler.end(),
      cell_primal      = dof_handler_primal.begin_active(),
      cell_adjoint     = dof_handler_adjoint.begin_active();
      for (; joint_cell!=joint_endc; ++joint_cell, ++cell_primal, ++cell_adjoint)
        {
          joint_cell->get_dof_indices (local_joint_dof_indices);
          cell_primal->get_dof_indices (local_dof_indices_primal);
          cell_adjoint->get_dof_indices (local_dof_indices_adjoint);

          for (unsigned int i=0; i<joint_fe.dofs_per_cell; ++i)
            if (joint_fe.system_to_base_index(i).first.first == 0)
              {
                Assert (joint_fe.system_to_base_index(i).second
                        <
                        local_dof_indices_primal.size(),
                        ExcInternalError());
                joint_solution(local_joint_dof_indices[i])
                  = solution_primal(local_dof_indices_primal[joint_fe.system_to_base_index(i).second]);
              }
            else
              {
                Assert (joint_fe.system_to_base_index(i).first.first == 1,
                        ExcInternalError());
                Assert (joint_fe.system_to_base_index(i).second
                        <
                        local_dof_indices_adjoint.size(),
                        ExcInternalError());
                joint_solution(local_joint_dof_indices[i])
                  = solution_adjoint(local_dof_indices_adjoint[joint_fe.system_to_base_index(i).second]);
              }
        }
    }



  std::vector<std::string> solution_names; 
  solution_names.push_back ("ux_primal");
  solution_names.push_back ("uy_primal"); 

  solution_names.push_back ("ux_adjoint");
  solution_names.push_back ("uy_adjoint"); 
   
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation
    (dim+dim, DataComponentInterpretation::component_is_scalar);


  DataOut<dim> data_out;
  data_out.attach_dof_handler (joint_dof_handler);  
   
  data_out.add_data_vector (joint_solution, solution_names,
			    DataOut<dim>::type_dof_data,
			    data_component_interpretation);
  
  data_out.build_patches ();

  std::string filename_basis;
  filename_basis  = "solution_nse_2d_"; 
   
  std::ostringstream filename;

  std::cout << "------------------" << std::endl;
  std::cout << "Write solution" << std::endl;
  std::cout << "------------------" << std::endl;
  std::cout << std::endl;
  filename << filename_basis
	   << Utilities::int_to_string (refinement_cycle, 5)
	   << ".vtk";
  
  std::ofstream output (filename.str().c_str());
  data_out.write_vtk (output);

}

// With help of this function, we extract 
// point values for a certain component from our
// discrete solution. 
template <int dim>
double Poisson_PU_DWR_Problem<dim>::compute_point_value (Point<dim> p, 
					       const unsigned int component) const  
{
 
  Vector<double> tmp_vector(dim);
  VectorTools::point_value (dof_handler_primal, 
			    solution_primal, 
			    p, 
			    tmp_vector);
  
  return tmp_vector(component);
}

// Now, we arrive at the function that is responsible 
// to compute the line integrals for the boundary evaluation. 
template <int dim>
void Poisson_PU_DWR_Problem<dim>::compute_boundary_tensor()
{
    
  const QGauss<dim-1> face_quadrature_formula (degree+2);
  FEFaceValues<dim> fe_face_values (fe_primal, face_quadrature_formula, 
				    update_values | update_gradients | update_normal_vectors | 
				    update_JxW_values);
  
  const unsigned int dofs_per_cell = fe_primal.dofs_per_cell;
  const unsigned int n_face_q_points = face_quadrature_formula.size();

  std::vector<unsigned int> local_dof_indices (dofs_per_cell);
  std::vector<Vector<double> >  face_solution_values (n_face_q_points, 
						      Vector<double> (dim));

  std::vector<std::vector<Tensor<1,dim> > > 
    face_solution_grads (n_face_q_points, std::vector<Tensor<1,dim> > (dim));
  
  Tensor<1,dim> value;
  
  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler_primal.begin_active(),
    endc = dof_handler_primal.end();

   for (; cell!=endc; ++cell)
     {

       for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
	 {
	   if (cell->face(face)->at_boundary() && 
	       cell->face(face)->boundary_id()==0)
	   {
	     fe_face_values.reinit (cell, face);
	     fe_face_values.get_function_values (solution_primal, face_solution_values);
	     fe_face_values.get_function_gradients (solution_primal, face_solution_grads);
	 	      
	     for (unsigned int q=0; q<n_face_q_points; ++q)
	       {
		 Tensor<2,dim> grad_u;
		 grad_u[0][0] = face_solution_grads[q][0][0];
		 grad_u[0][1] = face_solution_grads[q][0][1];
		 grad_u[1][0] = face_solution_grads[q][1][0];
		 grad_u[1][1] = face_solution_grads[q][1][1];

		 value -= grad_u * 
		   fe_face_values.normal_vector(q) * fe_face_values.JxW(q); 
		 
	       }
	   } // end boundary id
       
	 } // end faces

     } // end element

   std::cout << "Face ux:   "  << "   " << std::setprecision(16) << value[0] << std::endl;
   std::cout << "Face uy:   "  << "   " << std::setprecision(16) << value[1] << std::endl;

   // TODO
   std::cout << "Aborting: No reference values yet computed." << std::endl;
   abort(); 
   reference_value = 0.0; // TODO (compute on sufficiently fine mesh or analytical value) 
 
   exact_error_local = 0.0;
   exact_error_local = std::abs(value[0] - reference_value);
   

}




// Here, we compute the two quantities of interest: 
// a boundary line integration
// a point evaluation
template<int dim>
void Poisson_PU_DWR_Problem<dim>::compute_functional_values()
{
  double u_point_value;
  
  u_point_value = compute_point_value(Point<dim>(0.5,0.5), 0); 

  reference_value_u_point = -7.3671353258859554e-02; // global ref 8

  if (adjoint_rhs == "point")
    {
      exact_error_local = 0.0;
      exact_error_local = std::abs(u_point_value - reference_value_u_point);
    }

  std::cout << "------------------" << std::endl;
  std::cout << "u(0.5,0.5):  "  << "   " << std::setprecision(16) << u_point_value << std::endl;
  std::cout << "------------------" << std::endl;
  
  // Compute boundary line integral
  // compute_boundary_tensor();

  std::cout << "------------------" << std::endl;
  
  std::cout << std::endl;
}


template<int dim>
void Poisson_PU_DWR_Problem<dim>::refine_mesh()
{
  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler_primal.begin_active(),
    endc = dof_handler_primal.end();
  
  for (; cell!=endc; ++cell)
    {
      // Uniform mesh refinement
      cell->set_refine_flag();
    }

  triangulation.execute_coarsening_and_refinement ();

}


////////////////////////////////////////////////////////////////////////////////////////////
// PU-DWR Error estimator


// Implementing a weak form of DWR localization using a partition-of-unity
// as it has been proposed in 
//
// T. Richter, T. Wick; 
// Variational Localizations of the Dual-Weighted Residual Estimator,
// Journal of Computational and Applied Mathematics,
// Vol. 279 (2015), pp. 192-208
//
// Some routines have been taken from step-14 in deal.II
template<int dim>
double Poisson_PU_DWR_Problem<dim>::compute_error_indicators_a_la_PU_DWR (const unsigned int refinement_cycle)
{

  // First, we re-initialize the error indicator vector that 
  // has the length of the space dimension of the PU-FE.
  // Therein we store the local errors at all degrees of freedom.
  // This is in contrast to usual procedures, where the error 
  // in general is stored cell-wise. 
  dof_handler_pou.distribute_dofs (fe_pou);
  error_indicators.reinit (dof_handler_pou.n_dofs());
  
  
  // Block 1 (building the dual weights):
  // In the following the very specific 
  // part (z-I_hz) of DWR is implemented.
  // This part is the same for classical error estimation
  // and PU error estimation.
  std::vector<unsigned int> block_component (2,0);
  
  DoFRenumbering::component_wise (dof_handler_adjoint, block_component);
  
  // Implement the interpolation operator
  // (z-z_h)=(z-I_hz)
  ConstraintMatrix dual_hanging_node_constraints;
  DoFTools::make_hanging_node_constraints (dof_handler_adjoint,
					   dual_hanging_node_constraints);
  dual_hanging_node_constraints.close();
  
  ConstraintMatrix primal_hanging_node_constraints;
  DoFTools::make_hanging_node_constraints (dof_handler_primal,
					   primal_hanging_node_constraints);
  primal_hanging_node_constraints.close();
  
  
  // Construct a local primal solution that 
  // has the length of the adjoint vector
  std::vector<unsigned int> dofs_per_block (1);
  DoFTools::count_dofs_per_block (dof_handler_adjoint, 
				  dofs_per_block, block_component);  
  const unsigned int n_u = dofs_per_block[0];
  
  BlockVector<double> solution_primal_of_adjoint_length;
  solution_primal_of_adjoint_length.reinit(1);
  solution_primal_of_adjoint_length.block(0).reinit(n_u);
  solution_primal_of_adjoint_length.collect_sizes ();
  
  // Main function 1: Interpolate cell-wise the 
  // primal solution into the dual FE space.
  // This rescaled primal solution is called
  //   ** solution_primal_of_adjoint_length **
  FETools::interpolate (dof_handler_primal,
			solution_primal,
			dof_handler_adjoint,
			dual_hanging_node_constraints,
			solution_primal_of_adjoint_length);
  
  
  
  // Local vectors of dual weights obtained
  // from the adjoint solution
  BlockVector<double> dual_weights;
  dual_weights.reinit(1);
  dual_weights.block(0).reinit(n_u);
  dual_weights.collect_sizes ();
  
  // Main function 2: Execute (z-I_hz) (in the dual space),
  // yielding the adjoint weights for error estimation.
  FETools::interpolation_difference (dof_handler_adjoint,
				     dual_hanging_node_constraints,
				     solution_adjoint,
				     dof_handler_primal,
				     primal_hanging_node_constraints,
				     dual_weights);
  
  // end Block 1
  
  
  // Block 2 (evaluating the PU-DWR):
  // The following function has a loop inside that 
  // goes over all cells to collect the error contributions,
  // and is the `heart' of the DWR method. Therein 
  // the specific equation of the error estimator is implemented.
  
  
  // Info: must be sufficiently high for adjoint evaluations
  QGauss<dim>   quadrature_formula(degree+4); 
  
  FEValues<dim> fe_values_pou (fe_pou, quadrature_formula,
			       update_values    |
			       update_quadrature_points  |
			       update_JxW_values |
			       update_gradients);
  
  
  FEValues<dim> fe_values_adjoint (fe_adjoint, quadrature_formula,
				   update_values    |
				   update_quadrature_points  |
				   update_JxW_values |
				   update_gradients);
  
  
  const unsigned int   dofs_per_cell = fe_values_pou.dofs_per_cell;
  const unsigned int   n_q_points    = fe_values_pou.n_quadrature_points;
  
  Vector<double>  local_err_ind (dofs_per_cell);
  
  std::vector<unsigned int> local_dof_indices (dofs_per_cell);
  
  const FEValuesExtractors::Vector displacements (0);
  const FEValuesExtractors::Scalar pou_extract (0);
   
   
   std::vector<Vector<double> > primal_cell_values (n_q_points, 
						    Vector<double>(dim));
   
   std::vector<std::vector<Tensor<1,dim> > > primal_cell_gradients (n_q_points, 
								    std::vector<Tensor<1,dim> > (dim));
   
   std::vector<Vector<double> > dual_weights_values(n_q_points, 
						     Vector<double>(dim));
   
   std::vector<std::vector<Tensor<1,dim> > > dual_weights_gradients (n_q_points, 
								     std::vector<Tensor<1,dim> > (dim));
   
   typename DoFHandler<dim>::active_cell_iterator
     cell = dof_handler_pou.begin_active(),
     endc = dof_handler_pou.end();
   
   typename DoFHandler<dim>::active_cell_iterator
     cell_adjoint = dof_handler_adjoint.begin_active();
   
   for ( ; cell!=endc; ++cell, ++cell_adjoint)
     {
       fe_values_pou.reinit (cell);
       fe_values_adjoint.reinit (cell_adjoint);
       
       local_err_ind = 0;
       
       
       // primal solution (cell residuals)
       // But we use the adjoint FE since we previously enlarged the 
       // primal solution to the length of the adjoint vector.
       fe_values_adjoint.get_function_values (solution_primal_of_adjoint_length,
					      primal_cell_values);
      
       fe_values_adjoint.get_function_gradients (solution_primal_of_adjoint_length,
						 primal_cell_gradients);
       
       // adjoint weights
       fe_values_adjoint.get_function_values (dual_weights,
					      dual_weights_values);
      
       fe_values_adjoint.get_function_gradients (dual_weights,
						 dual_weights_gradients);
       
       
       
       
      // Gather local error indicators while running 
      // of the degrees of freedom of the partition of unity
      // and corresponding quadrature points.
       for (unsigned int q=0; q<n_q_points; ++q)
	 {
	   // Right hand side 
	  Tensor<1,dim> force;
	  force.clear();
	  force[0] = force_x;
	  force[1] = force_y;	  
	  
	  // Primal element values
	  Tensor<1,2> u;
	  u.clear();
	  u[0] = primal_cell_values[q](0);
	  u[1] = primal_cell_values[q](1);
	  
	  Tensor<2,dim> grad_u;
	  grad_u[0][0] = primal_cell_gradients[q][0][0];
	  grad_u[0][1] = primal_cell_gradients[q][0][1];
	  grad_u[1][0] = primal_cell_gradients[q][1][0];
	  grad_u[1][1] = primal_cell_gradients[q][1][1];
	  
	  
	  // Adjoint weights
	  Tensor<1,dim> dw_u;
	  dw_u[0] = dual_weights_values[q](0);
	  dw_u[1] = dual_weights_values[q](1);
	  
	  Tensor<2,dim> grad_dw_u;
	  grad_dw_u[0][0] = dual_weights_gradients[q][0][0];
	  grad_dw_u[0][1] = dual_weights_gradients[q][0][1];
	  grad_dw_u[1][0] = dual_weights_gradients[q][1][0];
	  grad_dw_u[1][1] = dual_weights_gradients[q][1][1];

	  
	  // Run over all PU degrees of freedom per cell (namely 4 DoFs for Q1 FE-PU)
	  for (unsigned int i=0; i<dofs_per_cell; ++i)
	    {
	      
	      Tensor<2,dim> grad_phi_psi;
	      grad_phi_psi[0][0] = fe_values_pou[pou_extract].value(i,q) * grad_dw_u[0][0] + dw_u[0] * fe_values_pou[pou_extract].gradient(i,q)[0];
	      grad_phi_psi[0][1] = fe_values_pou[pou_extract].value(i,q) * grad_dw_u[0][1] + dw_u[0] * fe_values_pou[pou_extract].gradient(i,q)[1];
	      grad_phi_psi[1][0] = fe_values_pou[pou_extract].value(i,q) * grad_dw_u[1][0] + dw_u[1] * fe_values_pou[pou_extract].gradient(i,q)[0];
	      grad_phi_psi[1][1] = fe_values_pou[pou_extract].value(i,q) * grad_dw_u[1][1] + dw_u[1] * fe_values_pou[pou_extract].gradient(i,q)[1];
	      
	    
	      
	      // Implement the error estimator
	      // J(u) - J(u_h) \approx \eta := (f,...) - (\nabla u, ...)
	      // 
	      // First part: (f,...)
	      local_err_ind(i) += (force  * dw_u * fe_values_pou[pou_extract].value(i,q)
				   ) * fe_values_pou.JxW (q);
	      
	      // Second part: - (\nabla u, ...)
	      local_err_ind(i) -= (scalar_product(grad_u, grad_phi_psi) 
				   ) * fe_values_pou.JxW (q);
	      

	    }

	 } // end q_points
       
      
       // Write all error contributions 
       // in their respective places in the global error vector.
       cell->get_dof_indices (local_dof_indices);
       for (unsigned int i=0; i<dofs_per_cell; ++i)
	 error_indicators(local_dof_indices[i]) += local_err_ind(i);
       
       
     } // end cell loop for PU FE elements
   
   
   // Finally, we eliminate and distribute hanging nodes in the error estimator
   ConstraintMatrix dual_hanging_node_constraints_pou;
   DoFTools::make_hanging_node_constraints (dof_handler_pou,
					    dual_hanging_node_constraints_pou);
   dual_hanging_node_constraints_pou.close();
   
   // Distributing the hanging nodes
   dual_hanging_node_constraints_pou.condense(error_indicators);
   
   // Averaging (making the 'solution' continuous)
   dual_hanging_node_constraints_pou.distribute(error_indicators);
   
   // end Block 2 
   
   
   // Block 3 (data and terminal print out)
   DataOut<dim> data_out;
   data_out.attach_dof_handler (dof_handler_pou);
   data_out.add_data_vector (error_indicators, "error_ind");
   data_out.build_patches ();	
   
   std::ostringstream filename;
   filename << "solution_error_indicators_"
	    << refinement_cycle
	    << ".vtk"
	    << std::ends;
   
   std::ofstream out (filename.str().c_str());
   data_out.write_vtk (out);
   
   
   // Print out on terminal
   std::cout << "------------------" << std::endl;
   std::cout << std::setiosflags(std::ios::scientific) << std::setprecision(2);
   std::cout << "   Dofs:                   " << dof_handler_primal.n_dofs()<< std::endl;
   std::cout << "   Exact error:            " << exact_error_local << std::endl; 
   double total_estimated_error = 0.0;
   for (unsigned int k=0;k<error_indicators.size();k++)
     total_estimated_error += error_indicators(k);
   
   // Take the absolute of the estimated error. 
   // However, we might check if the signs
   // of the exact error and the estimated error are the same.
   total_estimated_error = std::abs(total_estimated_error);
   
   std::cout << "   Estimated error (prim): " << total_estimated_error << std::endl; 
   
   // From Richter/Wick, JCAM, 2015 paper: compute indicator indices to check
   // effectivity of error estimator.
   double total_estimated_error_absolute_values = 0.0;
   for (unsigned int k=0;k<error_indicators.size();k++)
     total_estimated_error_absolute_values += std::abs(error_indicators(k));
   
   std::cout << "   Estimated error (abs):  " << total_estimated_error_absolute_values << std::endl; 
   
   std::cout << "   Ieff:                   " << total_estimated_error/exact_error_local << std::endl; 
   std::cout << "   Iind:                   " << total_estimated_error_absolute_values/exact_error_local << std::endl; 
   
   
   
   // Write everything into a file
   //file.precision(3);
   file << std::setiosflags(std::ios::scientific) << std::setprecision(2);
   file << dof_handler_primal.n_dofs() << "\t";
   file << exact_error_local << "\t";
   file << total_estimated_error << "\t";
   file << total_estimated_error_absolute_values << "\t";
   file << total_estimated_error/exact_error_local << "\t";
   file << total_estimated_error_absolute_values/exact_error_local << "\n";
   file.flush();
   
   // Write everything into a file gnuplot
   file_gnuplot << std::setiosflags(std::ios::scientific) << std::setprecision(2);
   //file_gnuplot.precision(3);
   file_gnuplot << dof_handler_primal.n_dofs() << "\t";
   file_gnuplot << exact_error_local << "\t";
   file_gnuplot << total_estimated_error << "\t";
   file_gnuplot << total_estimated_error_absolute_values << "\n";
   file_gnuplot.flush();
   
   
   // end Block 3


   // Block 4
   return total_estimated_error;
   

   // end Block 4
   
}


// Refinement strategy and carrying out the actual refinement.
template<int dim>
double Poisson_PU_DWR_Problem<dim>::refine_average_with_PU_DWR (const unsigned int refinement_cycle)
{
  // Step 1:
  // Obtain error indicators from PU DWR estimator
  double estimated_DWR_error = compute_error_indicators_a_la_PU_DWR (refinement_cycle);
  
  // Step 2: Choosing refinement strategy
  // Here: averaged refinement
  // Alternatives are in deal.II:
  // refine_and_coarsen_fixed_fraction for example
  for (Vector<float>::iterator i=error_indicators.begin();
       i != error_indicators.end(); ++i)
    *i = std::fabs (*i);


  const unsigned int  dofs_per_cell_pou   = fe_pou.dofs_per_cell;
  std::vector< unsigned int >  local_dof_indices(dofs_per_cell_pou);
  
  // Refining all cells that have values above the mean value
  double error_indicator_mean_value = error_indicators.mean_value();
  
  // Pou cell  later
  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler_pou.begin_active(),
    endc = dof_handler_pou.end();

  double error_ind = 0.0;
  //1.1; for drag and lift and none mesh smoothing
  //5.0; for pressure difference and maximal mesh smoothing
  double alpha = 1.1; 


  for (; cell!=endc; ++cell)
    {
      error_ind = 0.0;
      cell->get_dof_indices(local_dof_indices);
      
      for (unsigned int i=0; i<dofs_per_cell_pou; ++i)
	{
	  error_ind += error_indicators(local_dof_indices[i]);
	}
      
      // For uniform (global) mesh refinement,
      // just comment the following line
      if (error_ind >  alpha * error_indicator_mean_value)
	cell->set_refine_flag();
    }


  triangulation.execute_coarsening_and_refinement ();

  return estimated_DWR_error;
  
}






// As usual, we have to call the run method. It handles
// the output stream to the terminal.
// Finally, we perform the refinement loop of 
// the solution process.
template <int dim>
void Poisson_PU_DWR_Problem<dim>::run () 
{  

  // We set runtime parameters to drive the problem.
  // These parameters could also be read from a parameter file that
  // can be handled by the ParameterHandler object (see step-19)
  set_runtime_parameters ();

  // Initialize degrees of freedom
  setup_system_primal ();
  setup_system_adjoint ();

  std::cout << "\n==============================" 
	    << "====================================="  << std::endl;
  std::cout << "Parameters\n" 
	    << "==========\n"
	    << "TOL primal Newton: "   <<  lower_bound_newton_residuum << "\n"
	    << "Max. ref. cycles:  "   <<  max_no_refinement_cycles << "\n"
	    << "Max. number DoFs:  "   <<  max_no_degrees_of_freedom << "\n"
	    << "TOL DWR estimator: "   <<  TOL_DWR_estimator << "\n" 
	    << "Goal functional:   "   <<  adjoint_rhs << "\n"
	    << std::endl;

 
  // Refinement loop
  for (unsigned int cycle=0; cycle<max_no_refinement_cycles; ++cycle)
    { 
      std::cout << "\n===============================" 
		<< "=====================================" 
		<< std::endl; 
      std::cout << "Refinement cycle " << cycle << ':' << std::endl;



      // Solve problems: primal and adjoint
      newton_iteration_primal ();
      if (refinement_strategy == 1)
	solve_adjoint ();  
	
      // Compute goal functional values: line integral or point evaluation
      std::cout << std::endl;
      compute_functional_values();

      // Write solutions into vtk
      output_results (cycle);
     
      // Mesh refinement
      if (cycle >= 0)
	{
	  // Use solution transfer to interpolate solution
	  // to the next mesh in order to have a better 
	  // initial guess for the next refinement level.
	  BlockVector<double> tmp_solution_primal;
	  tmp_solution_primal = solution_primal;
	  
	  SolutionTransfer<dim,BlockVector<double> > solution_transfer (dof_handler_primal);
	  solution_transfer.prepare_for_coarsening_and_refinement(tmp_solution_primal);

	  // Choose refinement strategy. The choice 
	  // of '1' will take the PU DWR estimator.
	  double estimated_DWR_error = 0.0;
	  if (refinement_strategy == 0)
	    triangulation.refine_global (1);
	  else if (refinement_strategy == 1)
	    estimated_DWR_error = refine_average_with_PU_DWR (cycle);	 
	  else if (refinement_strategy == 2)
	    refine_mesh();
	  else 
	    {
	      std::cout << "Aborting: No such refinement strategy." << std::endl;
	      abort();
	    }

	  // A practical stopping criterion. Once
	  // the a posteriori error estimator (here PU DWR), has 
	  // been shown to be reliable, we can use the estimated 
	  // error as stopping criterion; say we want to 
	  // estimate the drag value up to a error tolerance of 1%
	  if (estimated_DWR_error < TOL_DWR_estimator)
	    {
	      std::cout << "Terminating. Goal functional has sufficient accuracy: \n"
			<< estimated_DWR_error
			<< std::endl;
	      break;
	    }
	  

	  // Update degrees of freedom after mesh refinement
	  if (cycle < max_no_refinement_cycles - 1)
	    {
	      std::cout << "\n------------------" << std::endl;
	      std::cout << "Setup DoFs for next refinement cycle:" << std::endl;

	      setup_system_primal();

	      if (dof_handler_primal.n_dofs() > max_no_degrees_of_freedom)
		{
		  // Set a sufficiently high number such that enough computations
		  // are done, but the memory of your machine / cluster is not exceeded.
		  std::cout << "Terminating because max number DoFs exceeded." << std::endl;
		  break;
		}

	      setup_system_adjoint();
	      
	      solution_transfer.interpolate(tmp_solution_primal, solution_primal); 
	    }


	} // end mesh refinement

    } // end refinement cycles
 
  
  
}

// The main function looks almost the same
// as in all other deal.II tuturial steps. 
int main () 
{
  try
    {
      deallog.depth_console (0);

      Poisson_PU_DWR_Problem<2> poisson_pu_dwr_problem(1);
      poisson_pu_dwr_problem.run ();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      
      return 1;
    }
  catch (...) 
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}




