#include <FrictionQPotFEM/UniformSingleLayer2d.h>
#include <GMatElastoPlasticQPot/Cartesian2d.h>
#include <GooseFEM/GooseFEM.h>
#include <cpppath.h>
#include <docopt/docopt.h>
#include <fmt/core.h>
#include <highfive/H5Easy.hpp>
#include <prrng.h>

#define MYASSERT(expr) MYASSERT_IMPL(expr, __FILE__, __LINE__)
#define MYASSERT_IMPL(expr, file, line) \
    if (!(expr)) { \
        throw std::runtime_error( \
            std::string(file) + ':' + std::to_string(line) + \
            ": assertion failed (" #expr ") \n\t"); \
    }

namespace FQF = FrictionQPotFEM::UniformSingleLayer2d;
namespace GM = GMatElastoPlasticQPot::Cartesian2d;

template <class T>
void CheckOrDumpWithDescription(
    H5Easy::File& file, const std::string& path, const T& data, const std::string& description)
{
    if (file.exist(path)) {
        T ret = H5Easy::load<T>(file, path);
        if (ret != data) {
            throw std::runtime_error("Mismatch: " + path);
        }
        return;
    }

    H5Easy::dump(file, path, data);
    H5Easy::dumpAttribute(file, path, "desc", description);
}

static const char USAGE[] =
    R"(RunFixedTop

Description:
    RunFixedTop simulation until maximum strain.

Usage:
    RunFixedTop [options] <data.hdf5>

Options:
    -h, --help      Show help.
        --version   Show version.

(c) Tom de Geus
)";

template <class T>
xt::xtensor<double, 2> read_epsy(const T& file, size_t N)
{
    auto initstate = H5Easy::load<xt::xtensor<uint64_t, 1>>(file, "/cusp/epsy/initstate");
    auto initseq = H5Easy::load<xt::xtensor<uint64_t, 1>>(file, "/cusp/epsy/initseq");
    auto eps_offset = H5Easy::load<double>(file, "/cusp/epsy/eps_offset");
    auto eps0 = H5Easy::load<double>(file, "/cusp/epsy/eps0");
    auto k = H5Easy::load<double>(file, "/cusp/epsy/k");
    auto nchunk = H5Easy::load<size_t>(file, "/cusp/epsy/nchunk");

    MYASSERT(initstate.size() == N);
    MYASSERT(initseq.size() == N);

    auto generators = prrng::auto_pcg32(initstate, initseq);

    auto epsy = generators.weibull({nchunk}, k);
    epsy *= (2.0 * eps0);
    epsy += eps_offset;
    epsy = xt::cumsum(epsy, 1);

    return epsy;
}

class Main : public FQF::System {

private:
    H5Easy::File m_file;
    GooseFEM::Iterate::StopList m_residual = GooseFEM::Iterate::StopList(20);
    size_t m_inc = 0;
    size_t m_iiter = 0;
    int m_kick = 1;
    double m_deps_kick;

public:
    Main(const std::string& fname) : m_file(fname, H5Easy::File::ReadWrite)
    {
        this->init(
            H5Easy::load<xt::xtensor<double, 2>>(m_file, "/coor"),
            H5Easy::load<xt::xtensor<size_t, 2>>(m_file, "/conn"),
            H5Easy::load<xt::xtensor<size_t, 2>>(m_file, "/dofs"),
            H5Easy::load<xt::xtensor<size_t, 1>>(m_file, "/iip"),
            H5Easy::load<xt::xtensor<size_t, 1>>(m_file, "/elastic/elem"),
            H5Easy::load<xt::xtensor<size_t, 1>>(m_file, "/cusp/elem"));

        this->setDt(H5Easy::load<double>(m_file, "/run/dt"));
        this->setMassMatrix(H5Easy::load<xt::xtensor<double, 1>>(m_file, "/rho"));
        this->setDampingMatrix(H5Easy::load<xt::xtensor<double, 1>>(m_file, "/damping/alpha"));

        this->setElastic(
            H5Easy::load<xt::xtensor<double, 1>>(m_file, "/elastic/K"),
            H5Easy::load<xt::xtensor<double, 1>>(m_file, "/elastic/G"));

        this->setPlastic(
            H5Easy::load<xt::xtensor<double, 1>>(m_file, "/cusp/K"),
            H5Easy::load<xt::xtensor<double, 1>>(m_file, "/cusp/G"),
            read_epsy(m_file, m_N));

        m_deps_kick = H5Easy::load<double>(m_file, "/run/epsd/kick");
    }

public:
    void run()
    {
        auto deps = FQF::version_dependencies();
        deps.push_back(prrng::version());

        CheckOrDumpWithDescription(
            m_file,
            "/meta/RunFixedBoundary/version",
            std::string(MYVERSION),
            "Code version at compile-time.");

        CheckOrDumpWithDescription(
            m_file,
            "/meta/RunFixedBoundary/version_dependencies",
            deps,
            "Library versions at compile-time.");

        if (m_file.exist("/meta/RunFixedBoundary/completed")) {
            fmt::print("Marked completed, skipping\n");
            return;
        }

        if (m_file.exist("/stored")) {
            size_t idx = H5Easy::getSize(m_file, "/stored") - std::size_t(1);
            m_inc = H5Easy::load<decltype(m_inc)>(m_file, "/stored", {idx});
            m_kick = H5Easy::load<decltype(m_kick)>(m_file, "/kick", {idx});
            m_t = H5Easy::load<decltype(m_t)>(m_file, "/t", {idx});
            this->setU(H5Easy::load<decltype(m_u)>(m_file, fmt::format("/disp/{0:d}", m_inc)));
            fmt::print("'{0:s}': Loading, inc = {1:d}\n", m_file.getName(), m_inc);
            m_kick = !m_kick;
        }
        else {
            H5Easy::dump(m_file, "/stored", 0, {0});
            H5Easy::dump(m_file, "/kick", 0, {0});
            H5Easy::dump(m_file, "/t", 0.0, {0});
            H5Easy::dump(m_file, fmt::format("/disp/{0:d}", m_inc), m_u);

            H5Easy::dumpAttribute(
                m_file, "/stored", "desc", std::string("List of increments in '/disp/{0:d}'"));

            H5Easy::dumpAttribute(
                m_file, "/kick", "desc", std::string("Per increment: triggered by kick or not"));

            H5Easy::dumpAttribute(
                m_file,
                "/t",
                "desc",
                std::string("Per increment: time at the end of the increment"));

            H5Easy::dumpAttribute(
                m_file,
                fmt::format("/disp/{0:d}", m_inc),
                "desc",
                std::string("Displacement at the end of the increment."));
        }

        for (++m_inc;; ++m_inc) {

            if (!m_material_plas.checkYieldBoundRight(5)) {
                CheckOrDumpWithDescription(
                    m_file,
                    "/meta/RunFixedBoundary/completed",
                    1,
                    "Signal that this program finished.");
                fmt::print("'{0:s}': Completed\n", m_file.getName());
                return;
            }

            this->addSimpleShearEventDriven(m_deps_kick, m_kick);

            if (!m_material_plas.checkYieldBoundRight(5)) {
                CheckOrDumpWithDescription(
                    m_file,
                    "/meta/RunFixedBoundary/completed",
                    1,
                    "Signal that this program finished.");
                fmt::print("'{0:s}': Completed\n", m_file.getName());
                return;
            }

            if (m_kick) {
                for (m_iiter = 0;; ++m_iiter) {

                    this->timeStep();

                    m_residual.roll_insert(this->residual());

                    if (m_residual.descending() && m_residual.all_less(1e-5)) {
                        break;
                    }

                    if (!m_material_plas.checkYieldBoundRight(5)) {
                        CheckOrDumpWithDescription(
                            m_file,
                            "/meta/RunFixedBoundary/completed",
                            1,
                            "Signal that this program finished.");
                        fmt::print("'{0:s}': Completed\n", m_file.getName());
                        return;
                    }
                }
            }

            fmt::print(
                "'{0:s}': inc = {1:8d}, kick = {2:1d}, iiter = {3:8d}\n",
                m_file.getName(),
                m_inc,
                m_kick,
                m_iiter);

            {
                H5Easy::dump(m_file, "/stored", m_inc, {m_inc});
                H5Easy::dump(m_file, "/kick", m_kick, {m_inc});
                H5Easy::dump(m_file, "/t", m_t, {m_inc});
                H5Easy::dump(m_file, fmt::format("/disp/{0:d}", m_inc), m_u);
            }

            m_iiter = 0;
            m_kick = !m_kick;
            this->quench();
            m_residual.reset();
        }
    }
};

int main(int argc, const char** argv)
{
    std::map<std::string, docopt::value> args =
        docopt::docopt(USAGE, {argv + 1, argv + argc}, true, std::string(MYVERSION));

    Main sim(args["<data.hdf5>"].asString());
    sim.run();

    return 0;
}
