#include <FrictionQPotFEM/UniformMultiLayerLeverDrive2d.h>
#include <prrng.h>
#include <GMatElastoPlasticQPot/Cartesian2d.h>
#include <GooseFEM/GooseFEM.h>
#include <highfive/H5Easy.hpp>
#include <fmt/core.h>
#include <cpppath.h>
#include <docopt/docopt.h>

#define MYASSERT(expr) MYASSERT_IMPL(expr, __FILE__, __LINE__)
#define MYASSERT_IMPL(expr, file, line) \
    if (!(expr)) { \
        throw std::runtime_error( \
            std::string(file) + ':' + std::to_string(line) + \
            ": assertion failed (" #expr ") \n\t"); \
    }

namespace FQF = FrictionQPotFEM::UniformMultiLayerLeverDrive2d;
namespace GM = GMatElastoPlasticQPot::Cartesian2d;

template <class T>
void CheckOrDumpWithDescription(
    H5Easy::File& file,
    const std::string& path,
    const T& data,
    const std::string& description)
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
    R"(RunFreeLever

Description:
    RunFreeLever simulation for shear history.

Usage:
    RunFreeLever [options] <data.hdf5>

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

public:

    Main(const std::string& fname) : m_file(fname, H5Easy::File::ReadWrite)
    {
        auto layers = H5Easy::load<xt::xtensor<size_t, 1>>(m_file, "/layers/stored");

        std::vector<xt::xtensor<size_t, 1>> elemmap;
        std::vector<xt::xtensor<size_t, 1>> nodemap;

        for (auto& layer : layers) {
            elemmap.push_back(H5Easy::load<xt::xtensor<size_t, 1>>(m_file, fmt::format("/layers/{0:d}/elemmap", layer)));
            nodemap.push_back(H5Easy::load<xt::xtensor<size_t, 1>>(m_file, fmt::format("/layers/{0:d}/nodemap", layer)));
        }

        this->init_lever(
            H5Easy::load<xt::xtensor<double, 2>>(m_file, "/coor"),
            H5Easy::load<xt::xtensor<size_t, 2>>(m_file, "/conn"),
            H5Easy::load<xt::xtensor<size_t, 2>>(m_file, "/dofs"),
            H5Easy::load<xt::xtensor<size_t, 1>>(m_file, "/iip"),
            elemmap,
            nodemap,
            H5Easy::load<xt::xtensor<bool, 1>>(m_file, "/layers/is_plastic"));

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

        this->layerSetTargetActive(H5Easy::load<xt::xtensor<bool, 2>>(m_file, "/drive/drive"));
        this->layerSetDriveStiffness(
            H5Easy::load<double>(m_file, "/drive/k"),
            static_cast<bool>(H5Easy::load<int>(m_file, "/drive/symmetric")));
        this->setLeverProperties(
            H5Easy::load<double>(m_file, "/drive/H"),
            H5Easy::load<xt::xtensor<double, 1>>(m_file, "/drive/height"));
    }

public:

    void run()
    {
        auto deps = FQF::version_dependencies();
        deps.push_back(prrng::version());

        CheckOrDumpWithDescription(m_file, "/meta/RunFreeLever/version", std::string(MYVERSION),
            "Code version at compile-time.");

        CheckOrDumpWithDescription(m_file, "/meta/RunFreeLever/version_dependencies", deps,
            "Library versions at compile-time.");

        if (m_file.exist("/meta/RunFreeLever/completed")) {
            fmt::print("Marked completed, skipping\n");
            return;
        }

        auto H = H5Easy::load<double>(m_file, "/drive/H");
        auto dgamma = H5Easy::load<xt::xtensor<double, 1>>(m_file, "/drive/delta_gamma");
        auto gamma = xt::cumsum(dgamma);
        size_t ninc = dgamma.size();
        size_t inc = 0;

        if (m_file.exist("/stored")) {
            size_t i = H5Easy::getSize(m_file, "/stored") - std::size_t(1);
            inc = H5Easy::load<decltype(inc)>(m_file, "/stored", {i});
            m_t = H5Easy::load<decltype(m_t)>(m_file, "/t", {i});
            this->setU(H5Easy::load<decltype(m_u)>(m_file, fmt::format("/disp/{0:d}", inc)));
            this->layerSetTargetUbar(H5Easy::load<xt::xtensor<double, 2>>(m_file, fmt::format("/drive/ubar/{0:d}", inc)));
            fmt::print("'{0:s}': Loading, inc = {1:d}\n", m_file.getName(), inc);
        }
        else {
            H5Easy::dump(m_file, "/stored", 0, {0});
            H5Easy::dump(m_file, "/t", 0.0, {0});
            H5Easy::dump(m_file, fmt::format("/disp/{0:d}", inc), m_u);
            H5Easy::dump(m_file, fmt::format("/drive/ubar/{0:d}", inc), this->layerTargetUbar());

            H5Easy::dumpAttribute(m_file, "/stored", "desc",
                std::string("List of increments in '/disp/{0:d}' and '/drive/ubar/{0:d}'"));

            H5Easy::dumpAttribute(m_file, "/t", "desc",
                std::string("Per increment: time at the end of the increment"));

            H5Easy::dumpAttribute(m_file, fmt::format("/disp/{0:d}", inc), "desc",
                std::string("Displacement (at the end of the increment)."));

            H5Easy::dumpAttribute(m_file, fmt::format("/drive/ubar/{0:d}", inc), "desc",
                std::string("Position of the loading frame of each layer."));
        }

        for (++inc; inc < ninc; ++inc) {

            this->setLeverTarget(H * gamma(inc));

            size_t iiter = this->minimise();

            fmt::print("'{0:s}': inc = {1:8d}, iiter = {2:8d}\n", m_file.getName(), inc, iiter);

            H5Easy::dump(m_file, "/stored", inc, {inc});
            H5Easy::dump(m_file, "/t", m_t, {inc});
            H5Easy::dump(m_file, fmt::format("/disp/{0:d}", inc), m_u);
            H5Easy::dump(m_file, fmt::format("/drive/ubar/{0:d}", inc), this->layerTargetUbar());
        }

        H5Easy::dump(m_file, "/meta/RunFreeLever/completed", 1);
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
